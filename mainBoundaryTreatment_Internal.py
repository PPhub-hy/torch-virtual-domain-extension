import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import numpy as np
import scipy.io as scio
import argparse
import gc

from BoundaryTreatment import *

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--out_name", help="name of this try")
args = parser.parse_args()

model_file = 'ComNS_Re100Ma2_200ep_LNO-Legendre_n4N12m6k2_interval50_norm0.5-0.5-5ln-10ln_Norm3-12-6-3-6-3_phy4_test3_model.pp'

network = torch.load('models/' + model_file)
network.eval()

L = 64
NG = L
NG_L = 0
NG_D = 64
NG_U = 64
NG_R = 10*(NG_D+NG_U)
Length_x = NG_L+NG_R
Length_y = NG_U+NG_D
delta_x = 2/128

xMin = -NG_L * delta_x
yMin = -NG_D * delta_x
xMax = xMin + Length_x * delta_x
yMax = yMin + Length_y * delta_x

u_lid = 1
alpha = 0

AUX_x = network.get_padding_R(Length_x+2*network.recept + 1)
AUX_y = network.get_padding_R(Length_y+2*network.recept + 1)
r = network.recept
print('length_x=',Length_x)
print('length_y=',Length_y)
print('AUX_x=', AUX_x)
print('AUX_y=', AUX_y)

Patchwise = False
mode = 'internal'
outputfilename = 'Poiseuille'
BoundaryAlgorithm = 'synchronous'

p_e = np.zeros((Length_y + 1, Length_x + 1, 2), dtype='float32')
for i in range(Length_y + 1):
    for j in range(Length_x + 1):
        p_e[i, j, 0] = j * delta_x + xMin
        p_e[i, j, 1] = i * delta_x + yMin

# 生成固壁几何文件
ShapeNum = 1
delta_s = 1/64
p_l0 = []
u_BC = []
seta = []
seta.append(np.pi)
for i in range(Length_x + 1):
    p_l0.append(p_e[-1, i, :])
    u_BC.append(0)
    seta.append(np.pi/2)
seta.pop()
seta.pop()
seta.append(0)
for i in range(Length_y - 1, -1, -1):
    p_l0.append(p_e[i, -1, :])
    u_BC.append(1 / 2 / 0.01 * 1.5 / (u_lid * (Length_y / 2)/0.01)/(Length_y / 2) * u_lid * u_lid * ((Length_y / 2) * (Length_y / 2) - (i - Length_y / 2)**2))
    seta.append(0)

for i in range(Length_x -1, -1, -1):
    p_l0.append(p_e[0, i, :])
    u_BC.append(0)
    seta.append(-np.pi/2)
seta.pop()
seta.append(-np.pi)
for i in range(1, Length_y + 1):
    p_l0.append(p_e[i, 0, :])
    u_BC.append(u_lid * (np.cos(np.pi/2*(i-Length_y/2)/(Length_y/2)))**2)
    seta.append(np.pi)
p_l0.pop()
p_l = [np.array(p_l0)]
u_BC.pop()
u_BC = torch.unsqueeze(torch.from_numpy(np.array(u_BC)).cuda(),1)
seta.pop()
seta = torch.from_numpy((np.array(seta)).reshape(len(seta),1)).cuda()
print('u_BC.shape',u_BC.shape)

p_ori0 = []
p_ori0.append(p_e[-1, 0, :])
p_ori0.append(p_e[-1, -1, :])
p_ori0.append(p_e[0, -1, :])
p_ori0.append(p_e[0, 0, :])
p_ori0.append(p_e[-1, 0, :])
p_ori = [np.array(p_ori0)]

p_e = p_e.reshape([(Length_y + 1) * (Length_x + 1), 2])

torch.cuda.synchronize()
t1=time.time()
BT = BoundaryTreatment(delta_x,delta_s,p_e,p_l,p_ori,seta,Length_x,Length_y,AUX_x,AUX_y,network,BoundaryAlgorithm,Patchwise=Patchwise,mode=mode,u_BC=u_BC)
torch.cuda.synchronize()
t2 = time.time()
print('Time cost for boundary treatment initialization is ',t2-t1)


u_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32')*u_lid*np.cos(alpha)
for i in range(Length_y + 1):
    u_NN[i, :] = u_lid * (np.cos(np.pi/2*(i-Length_y/2)/(Length_y/2)))**2
    '''u_NN[i, -1] = 1 / 2 / 0.01 * 1.5 / (u_lid * (Length_y / 2)/0.01)/(Length_y / 2) * u_lid * u_lid * (
                (Length_y / 2) * (Length_y / 2) - (i - Length_y / 2)**2)'''
v_NN = np.ones((Length_y + 1, Length_x + 1), dtype='float32')*u_lid*np.sin(alpha)
rho_NN = np.zeros((Length_y + 1, Length_x + 1), dtype='float32')
T_NN = np.zeros((Length_y + 1, Length_x + 1), dtype='float32')

u_NN = torch.from_numpy(u_NN).cuda()
v_NN = torch.from_numpy(v_NN).cuda()
rho_NN = torch.from_numpy(rho_NN).cuda()
T_NN = torch.from_numpy(T_NN).cuda()

u_NN1=torch.from_numpy(np.ones((Length_y + 1, Length_x + 1), dtype='float32')*0.6).cuda()
input = torch.stack((u_NN1, v_NN, rho_NN, T_NN))
input = torch.unsqueeze(input, 0)
input = F.pad(input,(r, r + AUX_x, r , r+ AUX_y), mode='replicate')
with torch.no_grad():
    output = network(input)
u_correction = torch.mean(output[0, 0, :, :])-0.6
v_correction = torch.mean(output[0, 1, :, :])
rho_correction = torch.mean(output[0, 2, :, :])
T_correction = torch.mean(output[0, 3, :, :])
print(u_correction,v_correction,rho_correction,T_correction)


cycle_num = 1000
for cycle in range(cycle_num+1):

    torch.cuda.synchronize()
    t1=time.time()

    u_NN = torch.unsqueeze(torch.unsqueeze(u_NN, 0), 0)
    v_NN = torch.unsqueeze(torch.unsqueeze(v_NN, 0), 0)
    rho_NN = torch.unsqueeze(torch.unsqueeze(rho_NN, 0), 0)
    T_NN = torch.unsqueeze(torch.unsqueeze(T_NN, 0), 0)

    u_NN = F.pad(u_NN, (r+AUX_x, r, 0, 0), mode='replicate')
    u_NN = F.pad(u_NN, (0, 0, r, r + AUX_y), mode='constant', value=0)
    v_NN = F.pad(v_NN, (r+AUX_x, r, 0, 0), mode='replicate')
    v_NN = F.pad(v_NN, (0, 0, r, r + AUX_y), mode='constant', value=0)

    # IBM
    rho_NN = F.pad(rho_NN, (r+AUX_x, r, r, r+ AUX_y), mode='replicate')
    T_NN = F.pad(T_NN, (r+AUX_x, r, r, r+ AUX_y), mode='replicate')
    
    # PS
    '''rho_NN = torch.exp(rho_NN) * torch.exp(T_NN)
    rho_NN = F.pad(rho_NN, (r, r+AUX_x, 0, 0), mode='replicate')
    rho_NN = F.pad(rho_NN, (0, 0, r , r+ AUX_y), mode='reflect')
    T_NN = F.pad(T_NN, (r, r+AUX_x, 0, 0), mode='replicate')
    T_NN = F.pad(T_NN, (0, 0, r, r+ AUX_y), mode='constant', value=0)
    rho_NN = torch.log(rho_NN / torch.exp(T_NN))'''


    input = torch.cat((u_NN, v_NN, rho_NN, T_NN), 1)
    with torch.no_grad():
        output = BT.network(input)[:, :, :-AUX_y, AUX_x:]

    boundary_u = torch.mm(BT.E2L, output[0, 0, :, :].reshape(-1,1)) - u_BC
    boundary_v = torch.mm(BT.E2L, output[0, 1, :, :].reshape(-1,1))

    u_NN = output[0, 0, :, :] - u_correction*0
    v_NN = output[0, 1, :, :] - v_correction*0
    rho_NN = output[0, 2, :, :] - rho_correction*0
    T_NN = output[0, 3, :, :] - T_correction *0
    boundary_u = boundary_u.cpu().detach().numpy()
    boundary_v = boundary_v.cpu().detach().numpy()

    torch.cuda.synchronize()
    t2 = time.time()

    if cycle%100==0:
        print(cycle, '/', cycle_num)
        scio.savemat('BoundaryTreatment/' + outputfilename + '_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                              'v': v_NN.cpu().detach().numpy(),
                                                              'rho': rho_NN.cpu().detach().numpy(),
                                                              'T': T_NN.cpu().detach().numpy(),
                                                                                               'boundary_u': boundary_u,
                                                                                               'boundary_v': boundary_v,
                                                                                               'u_BC':u_BC.cpu().detach().numpy(),
                                                                                               'p_l':p_l[0],
                                                                                               'seta':seta.cpu().detach().numpy()
                                                                                               })

    u_NN[:, 0] = torch.from_numpy(u_lid * (np.cos(
        np.pi / 2 * (np.array([i for i in range(Length_y+1)]) - Length_y / 2) / (Length_y / 2))) ** 2).cuda()
    u_NN[:, -1] = torch.from_numpy(1 / 2 / 0.01 * 1.5 /(u_lid * (Length_y / 2)/0.01)/(Length_y / 2)*u_lid*u_lid  * (
                (Length_y / 2) * (Length_y / 2) - (np.array([i for i in range(Length_y+1)]) - Length_y / 2)**2)).cuda()

    u_NN[0, :] = 0
    u_NN[-1, :] = 0
    v_NN[0, :] = 0
    v_NN[-1, :] = 0
    v_NN[:, 0] = 0
    rho_NN[:, 0] = 0
    T_NN[0, :] = 0
    T_NN[-1, :] = 0
    T_NN[:, 0] = 0
    gc.collect()

scio.savemat('BoundaryTreatment/loss_' + outputfilename + '.mat', mdict={'loss': np.array(BT.loss_his),
                                                                             'input_solid_init':BT.input_solid_init,
                                                                             'input_solid_coeff':BT.input_solid_coeff,
                                                                             'loss_coeff':BT.loss_coeff,
                                                                             'optimizer_option':BT.optimizer_option,
                                                                             'momentum':BT.momentum,
                                                                             'lr':BT.lr,
                                                                             'round':np.array(BT.round_his)})




