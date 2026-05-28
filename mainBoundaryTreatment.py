import torch
import numpy as np
import scipy.io as scio
import gc
import torch.nn.functional as F
import argparse
import math

from BoundaryTreatment import *

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--out_name", help="name of this try")
args = parser.parse_args()


model_file = 'ComNS_Re100Ma2_200ep_LNO-Legendre_n4N12m6k2_interval50_norm0.5-0.5-5ln-10ln_Norm3-12-6-3-6-3_phy4_test3_model.pp'

network = torch.load('models/' + model_file)
network.eval()

N = 12

L =64
NG = L

'''CCD10'''
NG_L = L*6
NG_D = L*6
NG_U = L*6
NG_R = L*12

'''Vehicle'''
'''NG_L = 8*L
NG_D = 0*L
NG_U = int(6.5*L)
NG_R = 50*L'''

Length_x = NG_L+NG_R
Length_y = NG_U+NG_D
delta_x = 2/128

outputfilename = 'CC'
#outputfilename = 'truck'
BoundaryAlgorithm = 'synchronous' # 'synchronous'
Patchwise = False
l_R = 2

u_lid = 1
u_rotor = 2

AUX_x = network.get_padding_R(Length_x+2*network.recept + 1)
AUX_y = network.get_padding_R(Length_y+2*network.recept)

r = network.recept


xMin = -NG_L*delta_x
xMax = xMin + delta_x * Length_x
yMin = -NG_D*delta_x
yMax = yMin + delta_x * Length_y
print('length_x=',Length_x)
print('length_y=',Length_y)
print('AUX_x=', AUX_x)
print('AUX_y=', AUX_y)
t_interval = 5
delta_t = 0.05
Re = 100
Ma = 0.2
R_fluid = 1/1.4/Ma/Ma
t = 0
alpha = -0/180*np.pi

'''计算IBM节点坐标'''
filename = 'geometry/CircularCylinderD10.mat'
#filename = 'geometry/BoundaryCurve_truck2.mat'


'''数据文件格式：
p_l:n*2数组 顺时针记录边界Lagrange点坐标，第一个点出现2次以实现封闭，从坐标原点开始
seta:p_l各点对应表面与x轴的夹角，第一个点只出现一遍
delta_s:Lagrange点之间的距离
p_b:p_l向外n格点坐标，用于计算法向速度梯度，第一个点只出现一遍'''
this_raw_data = scio.loadmat(filename)
ShapeNum = int(this_raw_data['ShapeNum'][0][0])
seta = torch.from_numpy(this_raw_data['seta']).cuda()
p_l = []
for i in range(ShapeNum):
    p_l.append(this_raw_data['p_l' + str(i)])
    p_l[i][:, 0] += xMin + NG_L * delta_x
    p_l[i][:, 1] += yMin + NG_D * delta_x

p_ori = []
for i in range(ShapeNum):
    p_ori.append(this_raw_data['p_ori' + str(i)])
    p_ori[i][:, 0] += xMin + NG_L * delta_x
    p_ori[i][:, 1] += yMin + NG_D * delta_x
delta_s = this_raw_data['delta_s']


p_e = np.zeros((Length_y, Length_x + 1, 2), dtype='float32')
for i in range(Length_y):
    for j in range(Length_x + 1):
        p_e[i, j, 0] = j * delta_x + xMin
        p_e[i, j, 1] = i * delta_x + yMin
p_e = p_e.reshape([(Length_y) * (Length_x + 1), 2])

for i in range(ShapeNum):
    if i == 0:
        # size of the geometry itself
        xMin_geo = np.min(p_l[i][:,0])
        xMax_geo = np.max(p_l[i][:,0])
        yMin_geo = np.min(p_l[i][:,1])
        yMax_geo = np.max(p_l[i][:,1])
    else:
        if np.min(p_l[i][:,0]) < xMin_geo:
            xMin_geo = np.min(p_l[i][:,0])
        if np.max(p_l[i][:,0]) > xMax_geo:
            xMax_geo = np.max(p_l[i][:,0])
        if np.min(p_l[i][:,1]) < yMin_geo:
            yMin_geo = np.min(p_l[i][:,1])
        if np.max(p_l[i][:,1]) > yMax_geo:
            yMax_geo = np.max(p_l[i][:,1])
# 划定需要进行反向传播的区域'
idx_x_start = max(((NG_L +int(64*xMin_geo)-N) // (N // 2)) * (N // 2), 0)
idx_x_end = min(((NG_L + int(64*xMax_geo)+N) // (N // 2) + 1) * (N // 2), Length_x)
idx_y_start = max(((NG_D + int(64*yMin_geo)-N) // (N // 2)) * (N // 2), 0)
idx_y_end = min(((NG_D + int(64*yMax_geo)+N) // (N // 2) + 1) * (N // 2), Length_y)


torch.cuda.synchronize()
t1=time.time()
BT = BoundaryTreatment(delta_x,delta_s,p_e,p_l,p_ori,seta,Length_x,Length_y-1,AUX_x,AUX_y,network,BoundaryAlgorithm,Patchwise=Patchwise,idx_x_start=idx_x_start, idx_x_end=idx_x_end, idx_y_start=idx_y_start, idx_y_end=idx_y_end)
torch.cuda.synchronize()
t2 = time.time()
print('Time cost for boundary treatment initialization is ',t2-t1)

'''x_record = int(((min(p_l[1][:, 0])+max(p_l[1][:, 0]))/2-xMin)/delta_x)
y_record_min = 100
y_record_max = 272'''

u_NN = np.ones((Length_y, Length_x + 1), dtype='float32')*u_lid*np.cos(alpha)
v_NN = np.ones((Length_y, Length_x + 1), dtype='float32')*u_lid*np.sin(alpha)
rho_NN = np.zeros((Length_y, Length_x + 1), dtype='float32')
T_NN = np.zeros((Length_y, Length_x + 1), dtype='float32')

u_NN = torch.from_numpy(u_NN).cuda()
v_NN = torch.from_numpy(v_NN).cuda()
rho_NN = torch.from_numpy(rho_NN).cuda()
T_NN = torch.from_numpy(T_NN).cuda()

input = torch.stack((u_NN, v_NN, rho_NN, T_NN))
input = torch.unsqueeze(input, 0)
input = F.pad(input,(r, r + AUX_x, r , r+ AUX_y), mode='replicate').to(torch.float32)
with torch.no_grad():
    output = network(input)
u_correction = torch.mean(output[0, 0, :, :])-u_lid
v_correction = torch.mean(output[0, 1, :, :])
rho_correction = torch.mean(output[0, 2, :, :])
T_correction = torch.mean(output[0, 3, :, :])
print(u_correction,v_correction,rho_correction,T_correction)

BT.loss_all = []


#with torch.no_grad():

output = torch.zeros(1, 4, Length_y, Length_x + 1).cuda()
#cycle_num = 192*7+1
cycle_num = 28

for cycle in range(1,cycle_num + 1):
    torch.cuda.synchronize()
    t1=time.time()
    input0 = torch.stack((u_NN, v_NN, rho_NN, T_NN))
    input0 = torch.unsqueeze(input0, 0)
    u_NN0 = u_NN

    '''CCD10'''
    u_NN = F.pad(u_NN, (r, r + AUX_x, r , r+ AUX_y), mode='constant', value=u_lid*np.cos(alpha))
    v_NN = F.pad(v_NN, (r, r + AUX_x, r , r+ AUX_y), mode='constant', value=u_lid*np.sin(alpha))
    rho_NN = F.pad(rho_NN, (r, r + AUX_x, r , r+ AUX_y), mode='constant', value=0)
    T_NN = F.pad(T_NN, (r, r + AUX_x, r , r+ AUX_y), mode='constant', value=0)
    input = torch.stack((u_NN, v_NN, rho_NN, T_NN))
    input = torch.unsqueeze(input, 0)

    '''Vehicle'''
    '''u_NN = F.pad(u_NN, (r, r+AUX_x, r, r+AUX_y), mode='constant', value=u_lid)
    v_NN = F.pad(v_NN, (r, r+AUX_x, r, r+AUX_y), mode='constant', value=0)
    input1 = torch.stack((u_NN, v_NN))
    input1 = torch.unsqueeze(input1, 0)
    input2 = torch.stack((rho_NN, T_NN))
    input2 = torch.unsqueeze(input2, 0)
    input2 = F.pad(input2, (r, r+AUX_x, 0, 0), mode='constant', value=0)
    input2 = F.pad(input2, (0, 0, r, r+AUX_y), mode='reflect')
    input = torch.cat((input1,input2),dim=1)'''


    if BoundaryAlgorithm == 'synchronous':
        '''NNB'''
        output, boundary_u, boundary_v = BT.NNBstep_part(input.to(torch.float32))
        u_NN = output[0, 0, :,:] - u_correction
        v_NN = output[0, 1, :, :] - v_correction
        rho_NN = output[0, 2, :, :] - rho_correction
        T_NN = output[0, 3, :, :] - T_correction
        boundary_u = boundary_u.cpu().detach().numpy()
        boundary_v = boundary_v.cpu().detach().numpy()





        if cycle == 28:
            print(cycle,'/',cycle_num)
            scio.savemat('BoundaryTreatment/' + outputfilename + '_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                                    'v': v_NN.cpu().detach().numpy(),
                                                                    'rho': rho_NN.cpu().detach().numpy(),
                                                                    'T': T_NN.cpu().detach().numpy(),
                                                                                                    'boundary_u':boundary_u,
                                                                                                    'boundary_v':boundary_v})

        '''rho = np.exp(rho_NN.detach().cpu().reshape(-1).numpy())
        rho_large = F.pad(rho_NN, (r, r + AUX_x, r, r + AUX_y), mode='constant', value=0)
        rho_large = np.exp(rho_large.detach().cpu().reshape(-1).numpy())
        T = np.exp(T_NN.detach().cpu().reshape(-1).numpy())
        T_large = F.pad(T_NN, (r, r + AUX_x, r, r + AUX_y), mode='constant', value=0)
        T_large = np.exp(T_large.detach().cpu().reshape(-1).numpy())             
        rho = torch.from_numpy(np.log(np.exp((BT.IsInPoly_input * rho_NN).cpu().detach().reshape(-1).numpy()) + (BT.InterMatrix * (rho_large * T_large))/T))
        rho_NN = rho.cuda().reshape([rho_NN.shape[0],rho_NN.shape[1]])'''


    if BoundaryAlgorithm == 'asynchronous':
        '''classic IBM'''
        with torch.no_grad():
            output = network(input[:, :, :, :])[:, :, :-AUX_y, :-AUX_x]
        u_NN = output[0, 0, :,:] - u_correction
        v_NN = output[0, 1, :, :] - v_correction
        rho_NN = output[0, 2, :, :] - rho_correction
        T_NN = output[0, 3, :, :] - T_correction
        loss_u, loss_v, boundary_u, boundary_v = BT.CalLoss(output)
        BT.loss_his.append([loss_u.item(), loss_v.item()])

    
        #if cycle==28 or cycle==16 or cycle==4:
        if cycle==60 or cycle==140 or cycle==280:
            print(cycle,'/',cycle_num)
            scio.savemat('BoundaryTreatment/' + outputfilename + '_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                                    'v': v_NN.cpu().detach().numpy(),
                                                                    'rho': rho_NN.cpu().detach().numpy(),
                                                                    'T': T_NN.cpu().detach().numpy(),
                                                                                                    'boundary_u':boundary_u,
                                                                                                    'boundary_v':boundary_v})

        u_NN, v_NN, rho_NN, T_NN = BT.ClassicIBMStep(output)
        u_NN = u_NN - u_correction
        v_NN = v_NN - v_correction
        rho_NN = rho_NN - rho_correction
        T_NN = T_NN - T_correction

        '''rho = np.exp(rho_NN.detach().cpu().reshape(-1).numpy())
        rho_large = F.pad(rho_NN, (r, r + AUX_x, r, r + AUX_y), mode='constant', value=0)
        rho_large = np.exp(rho_large.detach().cpu().reshape(-1).numpy())
        T = np.exp(T_NN.detach().cpu().reshape(-1).numpy())
        T_large = F.pad(T_NN, (r, r + AUX_x, r, r + AUX_y), mode='constant', value=0)
        T_large = np.exp(T_large.detach().cpu().reshape(-1).numpy())                   
        rho = torch.from_numpy(np.log(BT.IsInPoly * rho + (BT.InterMatrix * (rho_large * T_large))/T))
        rho_NN = rho.cuda().reshape([rho_NN.shape[0],rho_NN.shape[1]])'''

    torch.cuda.synchronize()
    t5 = time.time()

    gc.collect()

scio.savemat('BoundaryTreatment/loss_' + outputfilename + '.mat', mdict={'loss': np.array(BT.loss_his),
                                                                        'loss_coeff':BT.loss_coeff,
                                                                        'optimizer_option':BT.optimizer_option,
                                                                        'momentum':BT.momentum,
                                                                        'lr':BT.lr,
                                                                        'multiplyIsInPoly':BT.multiplyIsInPoly,
                                                                        'round':np.array(BT.round_his)})




