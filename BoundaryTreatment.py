import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
import math
import scipy.io as scio
import time

class BoundaryTreatment:
    def __init__(self,h,delta_s,p_e,p_l,p_ori,seta,Length_x,Length_y,AUX_x,AUX_y,network,BoundaryAlgorithm,Patchwise=False,mode='external',u_BC=0,v_BC=0, idx_x_start=0, idx_x_end=1, idx_y_start=0, idx_y_end=1):
        self.lr = 0.11
        self.momentum = 0.9
        self.input_solid_init = 0
        self.input_solid_coeff = 1
        self.loss_coeff = 1
        self.rounds = 1
        self.optimizer_option = 'Adam'
        self.multiplyIsInPoly = True
        self.loss_his = []
        self.round_his = []

        self.h = h
        self.delta_s = delta_s
        self.p_e = p_e
        self.p_l = p_l
        self.p_ori = p_ori # p_ori与p_l基本相同，唯一的区别是p_l较密的等间距分布，p_ori只记录转折点以加快代码判断速度
        self.ShapeNum = len(self.p_l)
        self.seta = seta

        self.network = network
        self.n = network.n
        self.k = network.k
        self.recept = network.recept

        self.BoundaryAlgorithm = BoundaryAlgorithm

        self.PatchLength_output = int(self.n / self.k * self.k) # 设置patch的output尺寸，必须是n/k的整数倍
        self.PatchLength = self.PatchLength_output + 2 * self.recept # patch的input尺寸

        self.Length_x = Length_x + AUX_x + 2 * self.recept
        self.Length_y = Length_y + AUX_y + 2 * self.recept

        self.AUX_x = AUX_x
        self.AUX_y = AUX_y

        self.PatchNum_x = int((Length_x + AUX_x + 1) / self.PatchLength_output)
        self.PatchNum_y = int((Length_y + AUX_y + 1) / self.PatchLength_output)

        self.xMin = np.min(p_e[:, 0]) - self.recept * self.h
        self.xMax = self.xMin + self.Length_x * self.h
        self.yMin = np.min(p_e[:, 1]) - self.recept * self.h
        self.yMax = self.yMin + self.Length_y * self.h

        self.xMax_l = np.zeros(self.ShapeNum)
        self.xMin_l = np.zeros(self.ShapeNum)
        self.yMax_l = np.zeros(self.ShapeNum)
        self.yMin_l = np.zeros(self.ShapeNum)
        self.Num_l = 0
        for ShapeOrder in range(self.ShapeNum):
            self.xMax_l[ShapeOrder] = np.max(p_l[ShapeOrder][:, 0])
            self.xMin_l[ShapeOrder] = np.min(p_l[ShapeOrder][:, 0])
            self.yMax_l[ShapeOrder] = np.max(p_l[ShapeOrder][:, 1])
            self.yMin_l[ShapeOrder] = np.min(p_l[ShapeOrder][:, 1])
            self.Num_l += self.p_l[ShapeOrder].shape[0]
        self.xMax_l_all = np.max(self.xMax_l)
        self.xMin_l_all = np.min(self.xMin_l)
        self.yMax_l_all = np.max(self.yMax_l)
        self.yMin_l_all = np.min(self.yMin_l)

        self.Patchwise = Patchwise
        self.mode = mode
        self.u_BC = u_BC
        self.v_BC = v_BC

        self.idx_x_start = idx_x_start
        self.idx_x_end = idx_x_end
        self.idx_y_start = idx_y_start
        self.idx_y_end = idx_y_end

        self.NNB_part = True
        if self.mode == 'internal':
            self.NNB_part = False

        if self.BoundaryAlgorithm == 'asynchronous':
            self.ClassicIBMGenerator()
            self.SymmetricGenerator()
            
        elif self.BoundaryAlgorithm == 'synchronous':
            self.NNBGenerator(Length_x,Length_y,AUX_x,AUX_y)
        if self.Patchwise:
            self.PatchWiseGenerator()
        


    '''Generators'''
    def ClassicIBMGenerator(self):
        print('Initializing classic immersed boundary method.')
        self.loss_his = []
        self.IsInPoly = np.ones(self.p_e.shape[0], dtype='float32') # 1代表在流体中，0代表在固体中
        self.IsAroundPoly = np.zeros(self.p_e.shape[0], dtype='float32') # 1代表在固壁附近
        for i in range(self.p_e.shape[0]):
            x = self.p_e[i, 0]
            y = self.p_e[i, 1]
            if self.xMin_l_all <= x <= self.xMax_l_all and self.yMin_l_all <= y <= self.yMax_l_all:
                for ShapeOrder in range(self.ShapeNum):
                    if self.mode == 'internal':
                        # for lid driven
                        '''if x<0 and y<0:
                            self.IsInPoly[i] = 0
                        else:
                            self.IsAroundPoly[i] = 1'''
                        # simplify for Poiseuille
                        if x == self.xMin_l[ShapeOrder] or x == self.xMax_l[ShapeOrder] or y == self.yMin_l[ShapeOrder] or y == self.yMax_l[ShapeOrder]:
                            self.IsAroundPoly[i] = 1
                        if self.xMin_l[ShapeOrder]< x < self.xMax_l[ShapeOrder] and self.yMin_l[ShapeOrder] < y < self.yMax_l[ShapeOrder]:
                            self.IsInPoly[i] = 0
                    else:
                        if self.xMin_l[ShapeOrder] - 2 * self.h <= x <= self.xMax_l[ShapeOrder] + 2 * self.h and self.yMin_l[ShapeOrder] - 2 * self.h <= y <= self.yMax_l[ShapeOrder] + 2 * self.h:
                            self.IsAroundPoly[i] = 1
                            if self.isPoiWithinPoly(self.p_e[i, :], self.p_ori[ShapeOrder]):
                                self.IsInPoly[i] = 0
        self.E2L, self.L2E = self.GetInterMatrix(self.p_e, self.p_l, self.Num_l, self.IsAroundPoly)
        if self.mode == 'internal':
            self.IsInPoly = 1 - self.IsInPoly
        print('Classic IBM generator completed.')

    def GetInterMatrix(self, p_e, p_l, Num_l, IsAroundPoly):
        E2L = sp.coo_matrix((Num_l, p_e.shape[0]), ['float32']).tolil()
        L2E = sp.coo_matrix((p_e.shape[0], Num_l), ['float32']).tolil()
        for j in range(p_e.shape[0]):
            if IsAroundPoly[j] == 0:
                continue
            tempNum = -1
            for ShapeOrder in range(self.ShapeNum):
                for i in range(p_l[ShapeOrder].shape[0]-1):
                    tempNum += 1
                    rx = abs(p_l[ShapeOrder][i, 0] - p_e[j, 0]) / self.h
                    if rx >= 2:
                        continue
                    ry = abs(p_l[ShapeOrder][i, 1] - p_e[j, 1]) / self.h
                    # 4-point
                    '''if rx < 2 and ry < 2:
                        if rx < 1:
                            dx = (3 - 2 * rx + np.sqrt(1 + 4 * rx - 4 * rx * rx)) / 8
                        elif rx < 2:
                            dx = (5 - 2 * rx - np.sqrt(-7 + 12 * rx - 4 * rx * rx)) / 8
                        if ry < 1:
                            dy = (3 - 2 * ry + np.sqrt(1 + 4 * ry - 4 * ry * ry)) / 8
                        elif ry < 2:
                            dy = (5 - 2 * ry - np.sqrt(-7 + 12 * ry - 4 * ry * ry)) / 8
                        E2L[tempNum, j] = dx * dy
                        L2E[j, tempNum] = dx * dy / self.h * self.delta_s'''
                    # 3-point
                    '''if rx < 1.5 and ry < 1.5:
                        if rx < 0.5:
                            dx = (1 + np.sqrt(-3 * rx * rx +1)) / 3
                        elif rx < 1.5:
                            dx = (5 - 3 * rx - np.sqrt(-3 * (1 - rx) * (1 - rx) + 1)) / 6
                        if ry < 0.5:
                            dy = (1 + np.sqrt(-3 * ry * ry +1)) / 3
                        elif ry < 1.5:
                            dy = (5 - 3 * ry - np.sqrt(-3 * (1 - ry) * (1 - ry) + 1)) / 6
                        E2L[tempNum, j] = dx * dy
                        L2E[j, tempNum] = dx * dy / self.h * self.delta_s'''

                    # 2-point
                    if rx < 1 and ry < 1:
                        if rx < 1:
                            dx = 1 - rx
                        if ry < 1:
                            dy = 1 - ry
                        E2L[tempNum, j] = dx * dy
                        L2E[j, tempNum] = dx * dy / self.h * self.delta_s

        E2L = E2L.tocsc().astype('float32')
        L2E = L2E.tocsc().astype('float32')
        return E2L, L2E

    def ClassicIBMStep(self, output):
        gamma = 1.4
        Ma = 0.2
        C_v = 1 / gamma / (gamma - 1) / Ma / Ma

        l_y = output.shape[2]
        l_x = output.shape[3]

        u = output[0, 0, :, :].detach().cpu().reshape(-1).numpy()
        v = output[0, 1, :, :].detach().cpu().reshape(-1).numpy()

        rho = np.exp(output[0, 2, :, :].detach().cpu().reshape(-1).numpy())

        RHOU = self.E2L.dot(rho * u)
        RHOV = self.E2L.dot(rho * v)


        fx = self.L2E.dot(-RHOU)
        fy = self.L2E.dot(-RHOV)
        delta_u = fx / rho
        delta_v = fy / rho
        u = (u + delta_u)  #* self.IsInPoly
        v = (v + delta_v)  #* self.IsInPoly
        return torch.from_numpy(u.reshape([l_y, l_x])).cuda(), \
               torch.from_numpy(v.reshape([l_y, l_x])).cuda(), \
               output[0, 2, :, :], \
               output[0, 3, :, :]

    def CalLoss(self, output):
        boundary_u = self.E2L.dot(output[0, 0, :, :].reshape(-1, 1).detach().cpu().numpy())
        boundary_v = self.E2L.dot(output[0, 1, :, :].reshape(-1, 1).detach().cpu().numpy())

        loss_u = np.sum(np.abs(boundary_u)) / self.Num_l
        loss_v = np.sum(np.abs(boundary_v)) / self.Num_l

        loss_n = torch.sum(torch.abs(boundary_u * torch.cos(self.seta) + boundary_v * torch.sin(self.seta))) / self.Num_l
        loss_t = torch.sum(torch.abs(boundary_u * torch.sin(self.seta) - boundary_v * torch.cos(self.seta))) / self.Num_l
        
        return loss_n, loss_t, boundary_u, boundary_v

    def SymmetricGenerator(self):
        # 获取对称处理矩阵，无论是整体或是patch都可

        if self.Patchwise:
            print('Initializing symmetric interpolation matrix (type: patch wise).')
            self.InterMatrix = []
            for i in range(len(self.batch_list)):
                p_e = []
                xx = self.poi_patch_list[i][0, 0] + self.h / 2
                yy = self.poi_patch_list[i][0, 1] + self.h / 2
                while yy < self.poi_patch_list[i][2, 1] - self.h / 2:
                    xx = self.poi_patch_list[i][0, 0] + self.h / 2
                    while xx < self.poi_patch_list[i][2, 0] - self.h / 2:
                        p_e.append(np.array([xx, yy]))
                        xx +=  self.h
                    yy += self.h
                assert len(p_e) == self.PatchLength ** 2
                p_e = np.array(p_e, dtype='float32')
                InterMatrix = self.GetSymmetricMatrix(p_e, self.p_l_boundary_patch_radial_list[i], self.IsInPoly_patch_list[i, 0, :, :].reshape(-1))
                self.InterMatrix.append(InterMatrix)
                if i % 20 == 0:
                    print('Patches generated: {}/{}'.format(i, len(self.batch_list)))
            # self.InterMatrix = np.array(self.InterMatrix, dtype='float32')

        else:
            print('Initializing symmetric interpolation matrix (type: whole domain).')
            '''self.IsInPoly = np.ones(self.p_e.shape[0], dtype='float32')  # 1代表在流体中，0代表在固体中
            for i in range(self.p_e.shape[0]):
                x = self.p_e[i, 0]
                y = self.p_e[i, 1]
                if self.xMin_l_all <= x <= self.xMax_l_all and self.yMin_l_all <= y <= self.yMax_l_all:
                    for ShapeOrder in range(self.ShapeNum):
                        if self.xMin_l[ShapeOrder] <= x <= self.xMax_l[ShapeOrder] and self.yMin_l[ShapeOrder] <= y <= \
                                self.yMax_l[ShapeOrder]:
                            if self.isPoiWithinPoly(self.p_e[i, :], self.p_l[ShapeOrder]):
                                self.IsInPoly[i] = 0'''
            self.InterMatrix = self.GetSymmetricMatrix(self.p_e, self.p_l, self.IsInPoly)

        print('Symmetric interpolation matrix generated.')


    def GetSymmetricMatrix(self, p_e, p_l, IsInPoly):
        # 输入网格点，固壁多边形，网格点是否在固壁内
        # 输出对称的插值矩阵
        InterMatrix = sp.coo_matrix((p_e.shape[0], (self.Length_x + 1) * (self.Length_y + 1)), ['float32']).tolil()
        for i in range(p_e.shape[0]):
            if IsInPoly[i] == 0:
                Order_closest, Order_closestshape = self.GetClosestLagrangePoint(p_e[i, :], p_l)
                assert Order_closest <= p_l[Order_closestshape].shape[0]
                BoundaryIntercept, ImagePoint, dist = self.GetBoundaryIntercept(p_e[i, :], p_l[Order_closestshape],
                                                                                Order_closest)
                coeff, order = self.GetBilinearInterpolation(ImagePoint)
                InterMatrix[i, order] = coeff

        return InterMatrix.tobsr().astype('float32')

    def GetClosestLagrangePoint(self, poi, p_l):
        # 输入：某点坐标，所在的固壁编号
        # 输出： 距该点最近的Lagrange点编号
        dist = 100000
        Order_closest = 0
        Order_closestshape = 0 # 具有最小距离的固壁多边形编号
        for ShapeOrder in range(len(p_l)):
            dist_temp = np.sqrt((poi[0] - p_l[ShapeOrder][:, 0]) ** 2 + (poi[1] - p_l[ShapeOrder][:, 1]) ** 2)
            if np.min(dist_temp) < dist:
                dist = np.min(dist_temp)
                Order_closest = np.argmin(dist_temp)
                Order_closestshape = ShapeOrder

        return Order_closest, Order_closestshape

    def GetBoundaryIntercept(self, poi, p_l, Order_closest):
        # 输入：固壁内点坐标，所在的固壁编号，最近的Lagrange点编号
        # 输出：Boundary intercept点的坐标， image point的坐标
        l = p_l.shape[0]
        poi_intersec1, InLine1, dist1 = self.GetVerticalIntersec(poi, p_l[(Order_closest - 1) % l, :], p_l[Order_closest, :])
        poi_intersec2, InLine2, dist2 = self.GetVerticalIntersec(poi, p_l[Order_closest, :], p_l[(Order_closest + 1) % l, :])

        if not (p_l[0, 0] == p_l[-1, 0] and p_l[0, 1] == p_l[-1, 1]):
            if Order_closest == 0:
                InLine1 = False
                dist1 = 100 * (self.xMax - self.xMin + self.yMax - self.yMin)
            elif Order_closest == p_l.shape[0] - 1:
                InLine2 = False
                dist2 = 100 * (self.xMax - self.xMin + self.yMax - self.yMin)
        if InLine1 and not InLine2:
            return poi_intersec1, 2 * poi_intersec1 - poi, dist1
        elif InLine2 and not InLine1:
            return poi_intersec2, 2 * poi_intersec2 - poi, dist2
        elif InLine1 and InLine2:
            if dist1 > dist2:
                return poi_intersec2, 2 * poi_intersec2 - poi, dist2
            else:
                return poi_intersec1, 2 * poi_intersec1 - poi, dist1
        else:
            return p_l[Order_closest, :], 2 * p_l[Order_closest, :] - poi, np.sqrt((p_l[Order_closest, :]-poi)**2)

    def GetVerticalIntersec(self, poi, s_poi, e_poi):
        # 输入： 点，线段起点与终点
        # 输出： 垂线交点坐标，是否落在线段内，垂直距离（若在线段内）
        poi_intersec = np.zeros(2, dtype='float32')
        dist = -1
        InLine = False
        if abs(s_poi[0] - e_poi[0]) < 1e-6:  # x=const 垂直线段
            poi_intersec[0] = s_poi[0]
            poi_intersec[1] = poi[1]
            if poi[1] >= min(s_poi[1], e_poi[1]) and poi[1] <= max(s_poi[1], e_poi[1]):
                InLine = True
                dist = abs(poi[0] - s_poi[0])
        elif abs(s_poi[1] - e_poi[1]) < 1e-6:  # y=const 水平线段
            poi_intersec[1] = s_poi[1]
            poi_intersec[0] = poi[0]
            if poi[0] >= min(s_poi[0], e_poi[0]) and poi[1] <= max(s_poi[0], e_poi[0]):
                InLine = True
                dist = abs(poi[1] - s_poi[1])
        else:
            k1 = (s_poi[1] - e_poi[1]) / (s_poi[0] - e_poi[0])
            b1 = e_poi[1] - k1 * e_poi[0]
            k2 = -1 / k1
            b2 = poi[1] - k2 * poi[0]
            poi_intersec[0] = -(b1 - b2) / (k1 - k2)
            poi_intersec[1] = k1 * poi_intersec[0] + b1
            if poi_intersec[0] >= min(s_poi[0], e_poi[0]) and poi_intersec[0] <= max(s_poi[0], e_poi[0]):
                InLine = True
                dist = np.sqrt(np.sum((poi_intersec - poi) ** 2))
        return poi_intersec, InLine, dist

    def GetBilinearInterpolation(self, poi):
        # 输入：某点坐标
        # 输出：该点在的四边形单元的双线性插值系数，单元四个顶点编号
        I = int((poi[0] - self.xMin) // self.h)
        J = int((poi[1] - self.yMin) // self.h)

        if I < 0:
            I = 0
            poi[0] = self.xMin
        if J < 0:
            J = 0
            poi[1] = self.yMin
        if I>self.Length_x:
            I = self.Length_x
            poi[0] = self.xMax
        if J > self.Length_y:
            J = self.Length_y
            poi[1] = self.yMax
        #assert I * self.h + self.xMin == self.p_e[I + (self.Length_x + 1) * J, 0]
        seta = ((poi[0]-self.xMin) / self.h - I) * 2 - 1
        ita = ((poi[1]-self.yMin) / self.h - J) * 2 - 1
        assert seta>=-1 and seta<=1
        assert ita>=-1 and ita<=1
        return np.array([(1 - seta) * (1 - ita) / 4, (1 + seta) * (1 - ita) / 4, (1 + seta) * (1 + ita) / 4,
                         (1 - seta) * (1 + ita) / 4]), np.array(
            [I + (self.Length_x + 1) * J, I + 1 + (self.Length_x + 1) * J, I + 1 + (self.Length_x + 1) * (J + 1),
             I + (self.Length_x + 1) * (J + 1)])

    def NNBGenerator(self, Length_x, Length_y, AUX_x, AUX_y):
        self.lr = 0.1
        self.momentum = 0.9
        self.input_solid_init = 0
        self.input_solid_coeff = 1
        self.loss_coeff = 1
        self.rounds = 200
        self.optimizer_option = 'Adam'
        self.multiplyIsInPolyforu = True
        self.multiplyIsInPolyforT = True


        self.avg_kernel_size = 5
        self.avg_filter = (torch.ones(4, 1, self.avg_kernel_size, self.avg_kernel_size) / self.avg_kernel_size / self.avg_kernel_size).cuda()
        self.loss_his = []
        self.round_his = []

        if self.Patchwise:
            print('Initializing neural network based optimization (type: patch wise).')
            self.E2L = []
            for i in range(len(self.batch_list)):
                p_e = []
                xx = self.poi_patch_output_list[i][0, 0] + self.h / 2
                yy = self.poi_patch_output_list[i][0, 1] + self.h / 2
                while yy < self.poi_patch_output_list[i][2, 1] - self.h / 2:
                    xx = self.poi_patch_output_list[i][0, 0] + self.h / 2
                    while xx < self.poi_patch_output_list[i][2, 0] - self.h / 2:
                        p_e.append(np.array([xx, yy]))
                        xx += self.h
                    yy += self.h
                assert len(p_e) == self.PatchLength_output ** 2
                p_e = np.array(p_e, dtype='float32')

                Num_l = 0
                for p_l in self.p_l_patch_radial_list[i]:
                    Num_l += len(p_l)
                E2L, L2E = self.GetInterMatrix(p_e, self.p_l_patch_radial_list[i], Num_l, [1 for _ in range(self.PatchLength_output ** 2)])
                self.E2L.append(torch.from_numpy(E2L.todense()).cuda())
                if i % 20 == 0:
                    print('Patches generated: {}/{}'.format(i, len(self.batch_list)))
        else:
            print('Initializing neural network based optimization (type: whole domain).')
            self.ClassicIBMGenerator()
            self.SymmetricGenerator()
            self.E2L = torch.from_numpy(self.E2L.todense()).cuda()
            #self.L2E = torch.from_numpy(self.L2E.todense()).cuda()
            self.IsInPoly = torch.from_numpy(self.IsInPoly).cuda().reshape(Length_y+1,Length_x+1)
            self.IsInPoly_input = self.IsInPoly
            self.IsInPoly = F.pad(self.IsInPoly, (self.recept, self.recept + AUX_x, self.recept, self.recept + AUX_y), mode='constant', value=1)
            if self.mode == 'internal':
                self.IsInPoly = 1 - self.IsInPoly
        
        if self.NNB_part:
            self.IsInPoly_part = self.IsInPoly[self.idx_y_start:self.idx_y_end + 2 * self.recept, self.idx_x_start:self.idx_x_end + 2 * self.recept]
            E2L_part = torch.zeros(self.Num_l, (self.idx_y_end - self.idx_y_start) * (self.idx_x_end - self.idx_x_start)).cuda()
            tempOrder = -1
            for ShapeOrder in range(self.ShapeNum):
                for i in range(self.p_l[ShapeOrder].shape[0]):
                    tempOrder += 1
                    poi = self.p_l[ShapeOrder][i, :]
                    idx_x = (poi[0] - np.min(self.p_e[:, 0])) / self.h - self.idx_x_start
                    idx_y = (poi[1] - np.min(self.p_e[:, 1])) / self.h - self.idx_y_start
                    # 4-point
                    '''for x in range(int(idx_x)-2, int(idx_x)+4):
                        r_x = abs(idx_x - x)
                        if r_x >= 2:
                            continue
                        elif r_x < 1:
                            d_x = 1/8 * (3-2*r_x+ math.sqrt(1+4*r_x-4*r_x*r_x))
                        else:
                            d_x = 1/8 * (5-2*r_x - math.sqrt(-7+12*r_x-4*r_x*r_x))
                        for y in range(int(idx_y)-2, int(idx_y)+4):
                            r_y = abs(idx_y - y)
                            if r_y >= 2:
                                continue
                            elif r_y < 1:
                                d_y = 1 / 8 * (3 - 2 * r_y + math.sqrt(1 + 4 * r_y - 4 * r_y * r_y))
                            else:
                                d_y = 1 / 8 * (5 - 2 * r_y - math.sqrt(-7 + 12 * r_y - 4 * r_y * r_y))'''
                            
                    # 2-point
                    for x in range(int(idx_x)-2, int(idx_x)+4):
                        r_x = abs(idx_x - x)
                        if r_x >= 1.5:
                            continue
                        for y in range(int(idx_y)-2, int(idx_y)+4):
                            r_y = abs(idx_y - y)
                            if r_y >= 2:
                                continue
                            if r_x < 1 and r_y < 1:
                                if r_x < 1:
                                    d_x = 1 - r_x
                                if r_y < 1:
                                    d_y = 1 - r_y
                                E2L_part[tempOrder, x + y * (self.idx_x_end - self.idx_x_start)] = d_x * d_y
            self.E2L_part = E2L_part


        # weighted loss test
        loss_weight = torch.from_numpy(np.ones((self.Num_l, 1), dtype='float32')).cuda()
        '''tempOrder = -1
        for ShapeOrder in range(self.ShapeNum):
                for i in range(self.p_l[ShapeOrder].shape[0]):
                    tempOrder += 1
                    if self.p_l[ShapeOrder][i, 0] > 4.9:
                        loss_weight[tempOrder] = 2'''
        self.loss_weight = loss_weight
        return 0

    def NNBstep(self, input):
        if not hasattr(self, 'input_solid') or self.rounds == 1:
            self.input_solid = torch.rand([input.shape[0], 4, input.shape[2], input.shape[3]]) * self.input_solid_init
            '''if not self.Patchwise and not hasattr(self, 'input_solid'):
                self.input_solid = torch.rand([input.shape[0], 4, input.shape[2], input.shape[3]]) * self.input_solid_init
            elif self.Patchwise and not hasattr(self, 'input_solid'):
                self.input_solid = torch.rand([1, 4, self.Length_y, self.Length_x]) * self.input_solid_init'''
            self.input_solid = torch.autograd.Variable(self.input_solid).cuda()
            self.input_solid.requires_grad = True
            self.input_solid.retain_grad()

        if self.optimizer_option == 'Adam':
            optimizer = optim.Adam([self.input_solid], lr=self.lr, weight_decay=2e-4)
        elif self.optimizer_option == 'SGD':
            optimizer = optim.SGD([self.input_solid], lr=self.lr,momentum=self.momentum, weight_decay=1e-4)
        elif self.optimizer_option == 'RMSprop':
            optimizer = optim.RMSprop([self.input_solid], lr=self.lr,momentum=self.momentum, weight_decay=1e-4)
        elif self.optimizer_option == 'LBFGS':
            optimizer = optim.LBFGS([self.input_solid], lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=-1)


        input_solid_avg = F.conv2d(self.input_solid, self.avg_filter, padding=int((self.avg_kernel_size - 1) / 2), groups=4)

        if self.Patchwise:
            round = 0
            while round < self.rounds:
                input_solid_avg_patch = self.ExtractPatchInput_nooutput(input_solid_avg)
                round += 1
                if self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :] * self.IsInPoly_patch_list + self.input_solid_coeff * input_solid_avg_patch[:, :2, :, :] * (1 - self.IsInPoly_patch_list),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :] * self.IsInPoly_patch_list), 1)
                elif not self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                    (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg_patch[:, :2, :, :] * (1 - self.IsInPoly_patch_list),
                     input[:, 2:3, :, :],
                     input[:, 3:, :, :]), 1)
                elif self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg_patch[:, :2, :, :] * (1 - self.IsInPoly_patch_list),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :] * self.IsInPoly_patch_list), 1)
                elif not self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg_patch[:, :2, :, :] * (1 - self.IsInPoly_patch_list),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :]), 1)

                output = self.network(this_input)
                loss_u = 0
                loss_v = 0
                for i in range(input.shape[0]):
                    boundary_u = torch.mm(self.E2L[i], output[i, 0, :, :].reshape(-1, 1)) - self.u_BC
                    boundary_v = torch.mm(self.E2L[i], output[i, 1, :, :].reshape(-1, 1)) - self.v_BC


                    loss_u += torch.mean(torch.abs(boundary_u))
                    loss_v += torch.mean(torch.abs(boundary_v))


                loss_boundary = (loss_u + loss_v) * self.loss_coeff

                optimizer.zero_grad()
                loss_boundary.backward()
                optimizer.step()
                self.loss_his.append([loss_u.item(), loss_v.item()])
                self.round_his.append(round)
                input_solid_avg = F.conv2d(self.input_solid, self.avg_filter, padding=int((self.avg_kernel_size - 1) / 2), groups=4)
                if round % 5 == 0:
                    print('BC optimized round {}: lr = {}, loss (u,v) = {}, {}'.format(round,optimizer.param_groups[0]['lr'], (loss_u).item(), (loss_v).item()))
                if round % 5 == 0:
                    scheduler.step()
        else:
            #input = input * self.IsInPoly
            round = 0
            loss_u = 1
            loss_v = 1
            #while (loss_u > 0.01 or loss_v > 0.01) and round < self.rounds:
            while round < self.rounds:
                #while (loss_u + loss_v)/2 > 0.05:
                #while (loss_u > 0.01 or loss_v > 0.01) and round < self.rounds:
                round += 1
                if self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :]* self.IsInPoly + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :]* self.IsInPoly), 1)
                elif not self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                    (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly),
                     input[:, 2:3, :, :],
                     input[:, 3:, :, :]), 1)
                elif self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :]* self.IsInPoly), 1)
                elif not self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                    this_input = torch.cat(
                        (input[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly),
                         input[:, 2:3, :, :],
                         input[:, 3:, :, :]), 1)

                output = self.network(this_input)[:, :, :-self.AUX_y, :-self.AUX_x]

                boundary_u = torch.mm(self.E2L, output[0, 0, :, :].reshape(-1,1)) - self.u_BC
                boundary_v = torch.mm(self.E2L, output[0, 1, :, :].reshape(-1,1)) - self.v_BC

                #print(boundary_u)
                loss_u = torch.sum(torch.abs(boundary_u)) / self.Num_l
                loss_v = torch.sum(torch.abs(boundary_v)) / self.Num_l

                #loss_boundary = (loss_u + loss_v) * self.loss_coeff*0.5
                #print((boundary_u * torch.cos(self.seta)).shape)
                loss_n = torch.sum(torch.abs(boundary_u * torch.cos(self.seta) + boundary_v * torch.sin(self.seta))) / self.Num_l
                loss_t = torch.sum(torch.abs(boundary_u * torch.sin(self.seta) - boundary_v * torch.cos(self.seta))) / self.Num_l
                self.omega_n = 4/5
                loss_boundary = loss_n * self.omega_n + loss_t * (1 - self.omega_n)


                optimizer.zero_grad()
                loss_boundary.backward()
                optimizer.step()

                input_solid_avg = F.conv2d(self.input_solid, self.avg_filter, padding=int((self.avg_kernel_size - 1) / 2), groups=4)
                self.loss_his.append([loss_u.item(),loss_v.item()])

                '''if round % 5 == 0:
                    print('BC optimized round {}: lr = {}, loss (u,v) = {}, {}'.format(round, optimizer.param_groups[0]['lr'], (loss_u).item(), (loss_v).item()))'''
                '''if round % 5 == 0:
                    scheduler.step()'''
        print('BC optimized round {}: lr = {}, loss (u,v) = {}, {}'.format(round, optimizer.param_groups[0]['lr'], (loss_u).item(), (loss_v).item()))
        output = output.detach()
        self.round_his.append(round)

        return output, boundary_u, boundary_v

    def NNBstep_part(self, input):
        assert self.Patchwise == False # use part NNB only when not use patchwise NNB
        assert self.NNB_part == True

        with torch.no_grad():
            output = self.network(input)[:, :, :-self.AUX_y, :-self.AUX_x]

        input_part = input[:, :, self.idx_y_start:self.idx_y_end + 2 * self.recept, self.idx_x_start:self.idx_x_end + 2 * self.recept]

        if not hasattr(self, 'input_solid'):
            self.input_solid = torch.rand([input_part.shape[0], 4, input_part.shape[2], input_part.shape[3]]) * self.input_solid_init
            self.input_solid = torch.autograd.Variable(self.input_solid).cuda()
            self.input_solid.requires_grad = True
            self.input_solid.retain_grad()

        if self.optimizer_option == 'Adam':
            optimizer = optim.Adam([self.input_solid], lr=self.lr, weight_decay=1e-4)
        elif self.optimizer_option == 'SGD':
            optimizer = optim.SGD([self.input_solid], lr=self.lr,momentum=self.momentum, weight_decay=1e-4)
        elif self.optimizer_option == 'RMSprop':
            optimizer = optim.RMSprop([self.input_solid], lr=self.lr,momentum=self.momentum, weight_decay=1e-4)
        elif self.optimizer_option == 'LBFGS':
            optimizer = optim.LBFGS([self.input_solid], lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=-1)


        input_solid_avg = F.conv2d(self.input_solid, self.avg_filter, padding=int((self.avg_kernel_size - 1) / 2), groups=4)

        
        #input = input * self.IsInPoly
        round = 0
        loss_u = 1
        loss_v = 1
        loss_u0 = 1
        loss_v0 = 1
        #while (loss_u > 0.01 or loss_v > 0.01) and round < self.rounds:
        while round < self.rounds:
            #while (loss_u + loss_v)/2 > 0.02:
            #while (loss_u > 0.01 or loss_v > 0.01) and round < self.rounds:
            round += 1
            if self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                this_input = torch.cat(
                    (input_part[:, :2, :, :]* self.IsInPoly_part + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly_part),
                        input_part[:, 2:3, :, :],
                        input_part[:, 3:, :, :]* self.IsInPoly_part), 1)
            elif not self.multiplyIsInPolyforT and self.multiplyIsInPolyforu:
                this_input = torch.cat(
                (input_part[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly_part),
                    input_part[:, 2:3, :, :],
                    input_part[:, 3:, :, :]), 1)
            elif self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                this_input = torch.cat(
                    (input_part[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly_part),
                        input_part[:, 2:3, :, :],
                        input_part[:, 3:, :, :]* self.IsInPoly_part), 1)
            elif not self.multiplyIsInPolyforT and not self.multiplyIsInPolyforu:
                this_input = torch.cat(
                    (input_part[:, :2, :, :] + self.input_solid_coeff * input_solid_avg[:, :2, :, :] * (1 - self.IsInPoly_part),
                        input_part[:, 2:3, :, :],
                        input_part[:, 3:, :, :]), 1)

            output_part = self.network(this_input)

            boundary_u = torch.mm(self.E2L_part, output_part[0, 0, :, :].reshape(-1,1)) - self.u_BC
            boundary_v = torch.mm(self.E2L_part, output_part[0, 1, :, :].reshape(-1,1)) - self.v_BC

            #print(boundary_u)
            loss_u = torch.sum(torch.abs(boundary_u)*self.loss_weight) / self.Num_l
            loss_v = torch.sum(torch.abs(boundary_v)*self.loss_weight) / self.Num_l
            #loss_boundary = (loss_u + loss_v) * self.loss_coeff

            #print((boundary_u * torch.cos(self.seta)).shape)
            loss_n = torch.sum(torch.abs(boundary_u * torch.cos(self.seta) + boundary_v * torch.sin(self.seta))) / self.Num_l
            loss_t = torch.sum(torch.abs(boundary_u * torch.sin(self.seta) - boundary_v * torch.cos(self.seta))) / self.Num_l
            self.omega_n = 4/5
            self.omega_t = 1 - self.omega_n
            loss_boundary = loss_n * self.omega_n + loss_t * self.omega_t

            optimizer.zero_grad()
            loss_boundary.backward()
            optimizer.step()

            input_solid_avg = F.conv2d(self.input_solid, self.avg_filter, padding=int((self.avg_kernel_size - 1) / 2), groups=4)
            self.loss_his.append([loss_n.item(),loss_t.item()])

            if round == 50:
                print('BC optimized round {}: lr = {}, loss (u,v) = {}, {}'.format(round, optimizer.param_groups[0]['lr'], (loss_t).item(), (loss_n).item()))
            '''if round % 5 == 0:
                scheduler.step()'''
        output[:, :, self.idx_y_start:self.idx_y_end, self.idx_x_start:self.idx_x_end] = output_part.detach()
        #output = output * self.IsInPoly_input
        print('BC optimized round {}: lr = {}, loss (u,v) = {}, {}'.format(round, optimizer.param_groups[0]['lr'], (loss_t).item(), (loss_n).item()))
        output = output.detach()
        self.round_his.append(round)

        return output, boundary_u, boundary_v

    def PatchWiseGenerator(self):

        print('Initializing patch-wise treatment.')

        self.poi_patch_list = []
        self.poi_patch_output_list = []
        self.batch_list = []
        self.p_l_patch_list = []
        self.p_l_patch_radial_list = []
        self.p_l_boundary_patch_radial_list = []
        self.IsInPoly_patch_list = []
        self.IsInPoly_patch_output_list = []

        for PatchOrder_x in range(self.PatchNum_x): # 可进一步优化，用xMax_l_all和xMin_l_all给出更小的range
            for PatchOrder_y in range(self.PatchNum_y):
                PatchOrder = PatchOrder_x + PatchOrder_y * self.PatchNum_x

                poi_patch = np.zeros([4, 2]) # patch顶点坐标
                poi_patch[0, 0] = (PatchOrder_x * self.PatchLength_output) * self.h +self.xMin -self.h / 2
                poi_patch[1, 0] = poi_patch[0, 0] + self.PatchLength * self.h + self.h
                poi_patch[2, 0] = poi_patch[1, 0]
                poi_patch[3, 0] = poi_patch[0, 0]
                poi_patch[0, 1] = (PatchOrder_y * self.PatchLength_output) * self.h +self.yMin -self.h / 2
                poi_patch[1, 1] = poi_patch[0, 1]
                poi_patch[2, 1] = poi_patch[0, 1] + self.PatchLength * self.h + self.h
                poi_patch[3, 1] = poi_patch[2, 1]

                poi_patch_output = np.zeros([4, 2])  # patch的输出的顶点坐标
                poi_patch_output[0, 0] = poi_patch[0, 0] + self.recept * self.h
                poi_patch_output[1, 0] = poi_patch[1, 0] - self.recept * self.h
                poi_patch_output[2, 0] = poi_patch_output[1, 0]
                poi_patch_output[3, 0] = poi_patch_output[0, 0]

                poi_patch_output[0, 1] = poi_patch[0, 1] + self.recept * self.h
                poi_patch_output[1, 1] = poi_patch_output[0, 1]
                poi_patch_output[2, 1] = poi_patch[2, 1] - self.recept * self.h
                poi_patch_output[3, 1] = poi_patch_output[2, 1]

                IsPatchIncludeBoundary = self.isPatchIncludeBoundary(poi_patch) # 判断patch是否包含边界
                for ShapeOrder in range(self.ShapeNum):
                    if IsPatchIncludeBoundary[ShapeOrder]:
                        IsPatchInFluid, IsInPoly_patch_output = self.isPatchPointInSolid(poi_patch_output, self.p_ori[ShapeOrder]) # 判断patch的output是否包含流体节点，若不包含则不需要计算
                        if not IsPatchInFluid:
                            IsPatchIncludeBoundary[ShapeOrder] = False

                if np.max(IsPatchIncludeBoundary):
                    self.GatherPatchInfo(PatchOrder,poi_patch,poi_patch_output,IsInPoly_patch_output,IsPatchIncludeBoundary)

        print('Patches require treatment gathered! The total number is ', len(self.batch_list))
        '''scio.savemat('BoundaryTreatment/test.mat', mdict={'p_l_list': np.array(self.p_l_patch_list),
                                                'p_l_radial_list': np.array(self.p_l_patch_radial_list),
                                                'poi_list': np.array(self.poi_patch_list)})'''


        for i in range(len(self.batch_list)):
            poi_patch = self.poi_patch_list[i]
            poi_patch_output = self.poi_patch_output_list[i]
            IsInPoly_patch_output = self.IsInPoly_patch_output_list[i]
            p_l_patch = self.p_l_patch_list[i]

            p_l_patch_radial,p_l_boundary_patch_radial,poi_base_left,poi_base_right = self.GetRadialSolidForPatch(poi_patch,poi_patch_output,IsInPoly_patch_output,p_l_patch)

            IsPatchInFluid = True
            IsInPoly_patch = [1 for _ in range(self.PatchLength ** 2)]
            for ShapeOrder in range(len(p_l_patch_radial)):
                IsPatchInFluid_temp, IsInPoly_patch_temp = self.isPatchPointInSolid(poi_patch,p_l_patch_radial[ShapeOrder])
                IsPatchInFluid = (IsPatchInFluid and IsPatchInFluid_temp)
                IsInPoly_patch = IsInPoly_patch and IsInPoly_patch_temp

            '''scio.savemat('BoundaryTreatment/patch_'+str(i)+'.mat',
                         mdict={'poi_patch': np.array(poi_patch),
                                'IsInPoly': np.array(IsInPoly_patch),
                                'p_l_patch_radial': np.array(p_l_patch_radial),
                                'p_l_patch': np.array(p_l_patch),
                                'poi_patch_output':np.array(poi_patch_output)})'''

            assert IsPatchInFluid == True

            self.p_l_patch_radial_list.append(p_l_patch_radial)
            self.p_l_boundary_patch_radial_list.append(p_l_boundary_patch_radial)
            self.IsInPoly_patch_list.append(IsInPoly_patch)
        print(torch.from_numpy(np.array(self.IsInPoly_patch_list, dtype='float32')).shape)
        self.IsInPoly_patch_list = torch.from_numpy(np.array(self.IsInPoly_patch_list, dtype='float32')).reshape([len(self.IsInPoly_patch_list), 1, self.PatchLength, self.PatchLength]).cuda()
        #self.IsInPoly_patch_list = self.IsInPoly_patch_list.permute(0,1,3,2)

        scio.savemat('BoundaryTreatment/test.mat',
                     mdict={'batch_list': np.array(self.batch_list),
                            'IsInPoly': np.array(IsInPoly_patch),
                            'p_l_patch_radial': np.array(p_l_patch_radial),
                            'p_l_patch': np.array(p_l_patch),
                            'poi_patch_output': np.array(poi_patch_output)})
        print('Patch-wise generator completed.')

    def ExtractPatchInput(self, input, output):
        # 从全场input中提取各个patch的input
        input_patch = torch.from_numpy(np.zeros([len(self.batch_list), 4, self.PatchLength, self.PatchLength], dtype='float32')).cuda()
        for i in range(len(self.batch_list)):
            PatchOrder = self.batch_list[i]
            PatchOrder_x = PatchOrder % self.PatchNum_x
            PatchOrder_y = PatchOrder // self.PatchNum_x
            input_patch[i, :, :, :] = input[0, :, PatchOrder_y*self.PatchLength_output:PatchOrder_y*self.PatchLength_output+self.PatchLength, PatchOrder_x*self.PatchLength_output: PatchOrder_x*self.PatchLength_output+self.PatchLength]
            output[0, 0:4, PatchOrder_y*self.PatchLength_output: (PatchOrder_y+1)*self.PatchLength_output, PatchOrder_x*self.PatchLength_output: (PatchOrder_x+1)*self.PatchLength_output] = 0
        return input_patch, output

    def ExtractPatchInput_nooutput(self, input):
        # 从全场input中提取各个patch的input
        input_patch = torch.from_numpy(np.zeros([len(self.batch_list), 4, self.PatchLength, self.PatchLength], dtype='float32')).cuda()
        for i in range(len(self.batch_list)):
            PatchOrder = self.batch_list[i]
            PatchOrder_x = PatchOrder % self.PatchNum_x
            PatchOrder_y = PatchOrder // self.PatchNum_x
            input_patch[i, :, :, :] = input[0, :, PatchOrder_y*self.PatchLength_output:PatchOrder_y*self.PatchLength_output+self.PatchLength, PatchOrder_x*self.PatchLength_output: PatchOrder_x*self.PatchLength_output+self.PatchLength]
        return input_patch

    def IntractPatchInput(self, output, output_patch):
        # 将边界处理完成的patch的output放回全场output中
        output_patch = torch.mul(output_patch, self.IsInPoly_patch_list[:, :, self.recept:-self.recept, self.recept:-self.recept])
        for i in range(len(self.batch_list)):
            PatchOrder = self.batch_list[i]
            PatchOrder_x = PatchOrder % self.PatchNum_x
            PatchOrder_y = PatchOrder // self.PatchNum_x
            #print(output.shape)
            #print(output_patch.shape)
            output[0, 0:4, PatchOrder_y*self.PatchLength_output: (PatchOrder_y+1)*self.PatchLength_output, PatchOrder_x*self.PatchLength_output: (PatchOrder_x+1)*self.PatchLength_output] += output_patch[i, 0:4, :, :]
        return output

    '''Intermediate functions'''
    def isPatchIncludeBoundary(self, poi_patch):
        # 判断patch中是否包含固壁边界，按ShapeOrder分别判断是否包含
        IsPatchIncludeBoundary = [False for _ in range(len(self.p_l))]

        if poi_patch[1,0] - self.h / 2<self.xMin_l_all or poi_patch[0,0] + self.h / 2>self.xMax_l_all or poi_patch[2,1] - self.h / 2<self.yMin_l_all or poi_patch[0,1] + self.h / 2>self.yMax_l_all:
            return IsPatchIncludeBoundary
        else:
            for ShapeOrder in range(self.ShapeNum):
                if poi_patch[1,0]<self.xMin_l[ShapeOrder] or poi_patch[0,0]>self.xMax_l[ShapeOrder] or poi_patch[2,1]<self.yMin_l[ShapeOrder] or poi_patch[0,1]>self.yMax_l[ShapeOrder]:
                    continue
                for i in range(len(self.p_l[ShapeOrder][:,0])):
                    if poi_patch[0,0]<=self.p_l[ShapeOrder][i,0]<=poi_patch[1,0] and poi_patch[0,1]<=self.p_l[ShapeOrder][i,1]<=poi_patch[2,1]:
                        IsPatchIncludeBoundary[ShapeOrder] = True
                        break
        return IsPatchIncludeBoundary

    def isRayIntersectsSegment(self, poi, s_poi, e_poi):
        if s_poi[1] == e_poi[1]:
            return False
        if s_poi[1] > poi[1] and e_poi[1] > poi[1]:
            return False
        if s_poi[1] < poi[1] and e_poi[1] < poi[1]:
            return False
        if s_poi[0] < poi[0] and e_poi[0] < poi[0]:
            return False
        if s_poi[1] == poi[1] and e_poi[1] > poi[1]:
            return False
        if e_poi[1] == poi[1] and s_poi[1] > poi[1]:
            return False

        xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])
        if xseg < poi[0]:
            return False
        else:
            return True

    def isPoiWithinPoly(self, poi, p_l):
        # 输入：点，多边形顶点数组
        # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组
        sinsc = 0  # 交点个数
        for i in range(len(p_l)-1):  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]

            if self.isRayIntersectsSegment(poi, p_l[i,:], p_l[i+1,:]):
                sinsc += 1  # 有交点就加1
        return True if sinsc % 2 == 1 else False


    def isPatchPointInSolid11(self, poi_patch,p_l):
        # 输入：patch顶点坐标，固壁顶点
        # 输出：patch是否在流体中，各点是否在固壁内
        IsInPoly = []
        IsPatchInFluid = False  # 标记patch的output是否有部分在流体中

        poi_x = poi_patch[0, 0] + self.h / 2
        poi_y = poi_patch[0, 1] + self.h / 2

        while poi_x < poi_patch[2, 0] - self.h / 2:
            while poi_y < poi_patch[2, 1] - self.h / 2:
                poi = [poi_x, poi_y]

                if self.isPoiWithinPoly(poi, p_l):
                    IsInPoly.append(0)
                else:
                    IsInPoly.append(1)
                    if not IsPatchInFluid:
                        IsPatchInFluid = True
                poi_y += self.h
            poi_y = poi_patch[0, 1] + self.h / 2
            poi_x += self.h

        return IsPatchInFluid, np.array(IsInPoly, dtype='float32')

    def isPatchPointInSolid(self, poi_patch,p_l):
        # 输入：patch顶点坐标，固壁顶点
        # 输出：patch是否在流体中，各点是否在固壁内
        IsInPoly = []
        IsPatchInFluid = False  # 标记patch的output是否有部分在流体中

        poi_x = poi_patch[0, 0] + self.h / 2
        poi_y = poi_patch[0, 1] + self.h / 2

        while poi_y < poi_patch[2, 1] - self.h / 2:
            while poi_x < poi_patch[2, 0] - self.h / 2:
                poi = [poi_x, poi_y]

                if self.isPoiWithinPoly(poi, p_l):
                    IsInPoly.append(0)
                else:
                    IsInPoly.append(1)
                    if not IsPatchInFluid:
                        IsPatchInFluid = True
                poi_x += self.h
            poi_x = poi_patch[0, 0] + self.h / 2
            poi_y += self.h

        return IsPatchInFluid, np.array(IsInPoly, dtype='float32')

    def GatherPatchInfo(self,PatchOrder,poi_patch,poi_patch_output,IsInPoly_patch_output,IsPatchIncludeBoundary):
        # 收集需要处理的patch的信息

        poi_intersect = []
        line_intersect = []
        order_intersect = []
        shape_intersect = []

        poi_intersect_output = []
        line_intersect_output = []
        order_intersect_output = []
        shape_intersect_output = []

        '''获取p_l与patch四边的交点'''
        for ShapeOrder in range(self.ShapeNum):
            if IsPatchIncludeBoundary[ShapeOrder]:
                poi, line, order = self.GetPatchLagrangeIntersect(poi_patch, self.p_l[ShapeOrder])
                poi_intersect.extend(poi)
                line_intersect.extend(line)
                order_intersect.extend(order)
                shape_intersect.extend([ShapeOrder for _ in range(len(line))])

                poi, line, order = self.GetPatchLagrangeIntersect(poi_patch_output, self.p_l[ShapeOrder])
                poi_intersect_output.extend(poi)
                line_intersect_output.extend(line)
                order_intersect_output.extend(order)
                shape_intersect_output.extend([ShapeOrder for _ in range(len(line))])
        poi_intersect = np.array(poi_intersect)
        poi_intersect_output = np.array(poi_intersect_output)
        len_poi_intersect = len(poi_intersect_output)
        assert len_poi_intersect == 4 or len_poi_intersect == 2 or len_poi_intersect == 0 # 暂不考虑多于两条交线的情况

        '''获取patch内固壁围成的多边形'''
        p_l_patch, p_l_patch_boundary = self.GetAllPolyInPatch(poi_patch,poi_intersect,line_intersect,order_intersect,shape_intersect)

        if len_poi_intersect == 2: # 只有一条交线的情形
            self.poi_patch_list.append(poi_patch)
            self.poi_patch_output_list.append(poi_patch_output)
            self.batch_list.append(PatchOrder)
            self.p_l_patch_list.append(p_l_patch)
            self.IsInPoly_patch_output_list.append(IsInPoly_patch_output)
        elif len_poi_intersect == 4: # 有两条交线的情形，需要判断流体域是否联通
            p_l_patch_output, p_l_patch_boundary_output = self.GetAllPolyInPatch(poi_patch_output,poi_intersect_output,line_intersect_output,order_intersect_output,shape_intersect_output)
            if len(p_l_patch_output) == 1: #若固壁区域只有一个，则流体域不联通，需拆成2个patch
                p_l_patch_output1, p_l_patch_boundary_output1 = self.GetAllPolyInPatch(poi_patch_output,poi_intersect_output[:2],line_intersect_output[:2],order_intersect_output[:2],shape_intersect_output[:2])
                p_l_patch_output2, p_l_patch_boundary_output2 = self.GetAllPolyInPatch(poi_patch_output,poi_intersect_output[2:],line_intersect_output[2:],order_intersect_output[2:],shape_intersect_output[2:])

                # 录入第一个patch
                IsPatchInFluid1, IsInPoly_patch_output1 = self.isPatchPointInSolid(poi_patch_output, p_l_patch_output1[0])
                if IsPatchInFluid1:
                    self.poi_patch_list.append(poi_patch)
                    self.poi_patch_output_list.append(poi_patch_output)
                    self.batch_list.append(PatchOrder)
                    self.p_l_patch_list.append(p_l_patch)
                    self.IsInPoly_patch_output_list.append(IsInPoly_patch_output1)

                # 录入第二个patch
                IsPatchInFluid2, IsInPoly_patch_output2 = self.isPatchPointInSolid(poi_patch_output, p_l_patch_output2[0])
                if IsPatchInFluid2:
                    self.poi_patch_list.append(poi_patch)
                    self.poi_patch_output_list.append(poi_patch_output)
                    self.batch_list.append(PatchOrder)
                    self.p_l_patch_list.append(p_l_patch)
                    self.IsInPoly_patch_output_list.append(IsInPoly_patch_output2)
                #print('IsPatchInFluid1=', IsPatchInFluid1)
                #print('IsPatchInFluid2=', IsPatchInFluid2)
                #print(self.isRayIntersectsSegment(np.array([0.1328,2.6172]), np.array([0.3828,2.372]), np.array([0.3828,3.008])))
                #print('len(p_l_patch2=),',len(p_l_patch2))
                #print(self.isPoiWithinPoly1(np.array([0.1328,2.6172]),p_l_patch2[0]))
                '''scio.savemat('BoundaryTreatment/test.mat',
                             mdict={'p_l_patch': p_l_patch,
                                    'p_l_patch1': p_l_patch1,
                                    'p_l_patch2': p_l_patch2,
                                    'poi_patch_output':poi_patch_output,
                                    'IsInPoly_patch_output1':IsInPoly_patch_output1,
                                    'IsInPoly_patch_output2':IsInPoly_patch_output2})'''

                #print(a)

                assert (IsPatchInFluid1 or IsPatchInFluid2) == True

            elif len(p_l_patch_output) == 2: #若固壁区域有两个，则流体域联通，处理方法同一条交线
                self.poi_patch_list.append(poi_patch)
                self.poi_patch_output_list.append(poi_patch_output)
                self.batch_list.append(PatchOrder)
                self.p_l_patch_list.append(p_l_patch)
                self.IsInPoly_patch_output_list.append(IsInPoly_patch_output)

        elif len_poi_intersect == 0:
            self.poi_patch_list.append(poi_patch)
            self.poi_patch_output_list.append(poi_patch_output)
            self.batch_list.append(PatchOrder)
            self.p_l_patch_list.append(p_l_patch)
            self.IsInPoly_patch_output_list.append(IsInPoly_patch_output)

    def GetPatchLagrangeIntersect(self, poi_patch, p_l):
        # 计算patch与固壁曲线的交点坐标，按进-出节点配对
        poi_intersect = []
        line_intersect = []
        order_intersect = []
        IsInPatch = False
        if poi_patch[0,0]<=p_l[0,0]<=poi_patch[1,0] and poi_patch[0,1]<=p_l[0,1]<=poi_patch[2,1]:
            IsInPatch = True

        for i in range(1, len(p_l[:,0])+1):
            ii = i
            if i> len(p_l[:,0])-1:
                ii = i-len(p_l[:,0])
            if (IsInPatch and not(poi_patch[0,0]<=p_l[ii,0]<=poi_patch[1,0] and poi_patch[0,1]<=p_l[ii,1]<=poi_patch[2,1])) \
                    or (not IsInPatch and (poi_patch[0,0]<=p_l[ii,0]<=poi_patch[1,0] and poi_patch[0,1]<=p_l[ii,1]<=poi_patch[2,1])):
                IsInPatch = not IsInPatch
                poi, line = self.isPatchIntersectLine(poi_patch, p_l[ii-1,:],p_l[ii,:])
                poi_intersect.append(poi)
                line_intersect.append(line)
                order_intersect.append(ii)
        #print('len(poi_intersect)=', len(poi_intersect))
        # 若第一个节点在patch内，则将最后一个交点调整为第一个交点
        if poi_patch[0, 0] <= p_l[0, 0] <= poi_patch[1, 0] and poi_patch[0, 1] <= p_l[0, 1] <= poi_patch[2, 1]:
            poi_intersect.insert(0, poi_intersect[-1])
            line_intersect.insert(0, line_intersect[-1])
            order_intersect.insert(0, order_intersect[-1])
            poi_intersect.pop()
            line_intersect.pop()
            order_intersect.pop()

        assert len(poi_intersect) % 2 == 0 # 暂不考虑交点为奇数的情况

        return poi_intersect, line_intersect, order_intersect

    def isTwoLineIntersect(self,s_poi0,e_poi0,s_poi1,e_poi1):
        # 输入：两线段的起点和终点坐标
        # 输出：两线段是否相交

        # 快速排斥实验
        if max(s_poi0[0],e_poi0[0])<min(s_poi1[0],e_poi1[0]) or max(s_poi0[1],e_poi0[1])<min(s_poi1[1],e_poi1[1]) or \
                max(s_poi1[0],e_poi1[0])<min(s_poi0[0],e_poi0[0]) or max(s_poi1[1],e_poi1[1])<min(s_poi0[1],e_poi0[1]):
            return False
        # 跨立实验：叉乘判断是否相交
        if ((s_poi0[0]-s_poi1[0])*(e_poi1[1]-s_poi1[1])-(s_poi0[1]-s_poi1[1])*(e_poi1[0]-s_poi1[0]))*\
                ((e_poi0[0]-s_poi1[0])*(e_poi1[1]-s_poi1[1])-(e_poi0[1]-s_poi1[1])*(e_poi1[0]-s_poi1[0]))>0 \
                or ((s_poi1[0]-s_poi0[0])*(e_poi0[1]-s_poi0[1])-(s_poi1[1]-s_poi0[1])*(e_poi0[0]-s_poi0[0]))\
                *((e_poi1[0]-s_poi0[0])*(e_poi0[1]-s_poi0[1])-(e_poi1[1]-s_poi0[1])*(e_poi0[0]-s_poi0[0]))>0:
            return False
        return True

    def isPatchIntersectLine(self,poi_patch,s_poi,e_poi):
        # 输入：patch顶点坐标，线段起点、终点坐标
        # 输出：交点坐标，交点所在边
        line_intersect = -1
        poi_intersect = np.zeros(2)

        # 判断特殊情况：固壁线段水平或竖直
        # 竖直
        if abs(s_poi[0] - e_poi[0]) < 1e-5:
            # 是否与边0相交
            if self.isTwoLineIntersect(poi_patch[0, :], poi_patch[1, :], s_poi, e_poi):
                line_intersect = 0
                poi_intersect[0] = s_poi[0]
                poi_intersect[1] = poi_patch[0, 1]
            # 是否与边2相交
            elif self.isTwoLineIntersect(poi_patch[2, :], poi_patch[3, :], s_poi, e_poi):
                line_intersect = 2
                poi_intersect[0] = s_poi[0]
                poi_intersect[1] = poi_patch[2, 1]
        # 水平
        elif abs(s_poi[1] - e_poi[1]) < 1e-5:
            # 是否与边1相交
            if self.isTwoLineIntersect(poi_patch[1, :], poi_patch[2, :], s_poi, e_poi):
                line_intersect = 1
                poi_intersect[0] = poi_patch[1, 0]
                poi_intersect[1] = s_poi[1]
            # 是否与边3相交
            elif self.isTwoLineIntersect(poi_patch[2, :], poi_patch[3, :], s_poi, e_poi):
                line_intersect = 3
                poi_intersect[0] = poi_patch[3, 0]
                poi_intersect[1] = s_poi[1]
        # 其余情形
        else:
            k = (e_poi[1] - s_poi[1]) / (e_poi[0] - s_poi[0])
            b = s_poi[1] - k * s_poi[0]
            if self.isTwoLineIntersect(poi_patch[0, :], poi_patch[1, :], s_poi, e_poi):
                line_intersect = 0
                poi_intersect = np.array([(poi_patch[0, 1] - b) / k, poi_patch[0, 1]])
            elif self.isTwoLineIntersect(poi_patch[1, :], poi_patch[2, :], s_poi, e_poi):
                line_intersect = 1
                poi_intersect = np.array([poi_patch[1, 0], k * poi_patch[1, 0] + b])
            elif self.isTwoLineIntersect(poi_patch[2, :], poi_patch[3, :], s_poi, e_poi):
                line_intersect = 2
                poi_intersect = np.array([(poi_patch[2, 1] - b) / k, poi_patch[2, 1]])
            elif self.isTwoLineIntersect(poi_patch[3, :], poi_patch[0, :], s_poi, e_poi):
                line_intersect = 3
                poi_intersect = np.array([poi_patch[0, 0], k * poi_patch[0, 0] + b])
        return poi_intersect, line_intersect

    def GetAllPolyInPatch(self, poi_patch,poi_intersect,line_intersect,order_intersect,shape_intersect):
        # 获取一个patch内所有固壁围成的多边形节点，可能有多个
        line_num_intersect = int(len(poi_intersect[:, 0]) / 2)

        '''获取固壁中在patch内的曲线部分（包含两端与patch边的交点）'''
        p_l_patch_boundary = []
        for i in range(line_num_intersect):
            p_l_temp = [poi_intersect[2*i,:]]
            if order_intersect[2*i]<= order_intersect[2*i+1]:
                for order in range(order_intersect[2*i], order_intersect[2*i+1]):
                    p_l_temp.append(self.p_l[shape_intersect[2*i]][order, :])
            else:
                for order in range(order_intersect[2 * i], len(self.p_l[shape_intersect[2*i]][:, 0])):
                    p_l_temp.append(self.p_l[shape_intersect[2 * i]][order, :])
                for order in range(order_intersect[2*i+1]):
                    p_l_temp.append(self.p_l[shape_intersect[2 * i]][order, :])

            p_l_temp.append(poi_intersect[2*i+1,:])
            p_l_patch_boundary.append(p_l_temp)

        p_l_patch,p_l_patch_boundary = self.Get_p_l_patch(poi_patch,poi_intersect,line_intersect,p_l_patch_boundary)

        return p_l_patch, p_l_patch_boundary

    def Get_p_l_patch(self,poi_patch,poi_intersect,line_intersect,p_l_patch_boundary):
        # 利用p_l_patch_boundary和poi_intersect获得patch内的完整solid多边形

        line_num_intersect = len(p_l_patch_boundary)

        poi_loop = []
        order_in_loop = []
        for i in range(4):
            poi_loop.append(poi_patch[i, :])
            order_in_loop.append(-1)
            flag = 0  # 标记一条边上是否有多个poi_intersect

            for j in range(len(poi_intersect[:, 0])):
                if line_intersect[j] == i:
                    flag += 1
                    poi_loop.append(poi_intersect[j, :])
                    order_in_loop.append(j)
            # 对同一边上的intersect点进行排序
            if flag > 1:
                for m in range(-flag, 0):
                    for n in range(m + 1, 0):

                        if (i == 0 and poi_loop[m][0] > poi_loop[n][0]) or (
                                i == 1 and poi_loop[m][1] > poi_loop[n][1]) or (
                                i == 2 and poi_loop[m][0] < poi_loop[n][0]) or (
                                i == 3 and poi_loop[m][1] < poi_loop[n][1]):
                            poi_loop[m], poi_loop[n] = poi_loop[n], poi_loop[m]
                            order_in_loop[m], order_in_loop[n] = order_in_loop[n], order_in_loop[m]

        p_l_patch = []
        flag = [True for _ in range(line_num_intersect)]
        for i in range(line_num_intersect):
            if flag[i]:
                p_l_temp = p_l_patch_boundary[i]
                flag[i] = False
                index = order_in_loop.index(2 * i + 1)

                while True:
                    index += 1
                    poi = self.LoopList(poi_loop, index)
                    # 若循环回到p_l开头，结束本次搜索
                    if poi[0] == poi_intersect[2 * i, 0] and poi[1] == poi_intersect[2 * i, 1]:
                        p_l_temp.append(self.LoopList(poi_loop, index))
                        break
                    # 若该节点为交线首点
                    if self.LoopList(order_in_loop, index) % 2 == 0:
                        line_order_intersect = self.LoopList(order_in_loop, index)
                        #p_l_temp.append(poi)
                        p_l_temp.extend(p_l_patch_boundary[line_order_intersect // 2])
                        flag[line_order_intersect // 2] = False

                        index = order_in_loop.index(line_order_intersect + 1)
                    # 若该节点为顶点
                    elif self.LoopList(order_in_loop, index) < 0:
                        p_l_temp.append(poi)
                p_l_patch.append(np.array(p_l_temp))

        return p_l_patch, p_l_patch_boundary

    def LoopList(self, List, order):
        # 循环获取list值
        if 0<= order < len(List):
            return List[order]
        else:
            return List[order % len(List)]

    def GetRadialSolidForPatch(self, poi_patch, poi_patch_output, IsInPoly_output, p_l_patch):
        # 输入：patch顶点，output顶点，output内节点是否在固壁内，patch内固壁多边形顶点
        # 输出：射线固壁顶点，射线固壁顶点在patch的部分

        p_l_patch_radial = []
        p_l_boundary_patch_radial = []


        for ShapeOrder in range(len(p_l_patch)):
            LeftTangentHandled = False
            RightTangentHandled = False
            generator_poi_base = self.GenerateFluidPoint(poi_patch_output, IsInPoly_output)
            while not (LeftTangentHandled and RightTangentHandled):
                poi_base = next(generator_poi_base)
                if poi_base == False:
                    if not LeftTangentHandled:
                        poi_tan_left = poi_tan_left_temp
                        order_tan_left = order_tan_left_temp
                        poi_base_left = poi_base_last
                    if not RightTangentHandled:
                        poi_tan_right = poi_tan_right_temp
                        order_tan_right = order_tan_right_temp
                        poi_base_right = poi_base_last
                    break

                #assert self.isPoiWithinPoly(poi_base, p_l_patch[ShapeOrder]) == False

                poi_tan_left_temp, poi_tan_right_temp, order_tan_left_temp, order_tan_right_temp = self.GetTangent(poi_base, p_l_patch[ShapeOrder])

                poi_base_last = poi_base

                if not LeftTangentHandled and order_tan_left_temp>=0:
                    # 判断切点是否在output内，若是，则换点
                    if (poi_patch_output[0,0]+self.h/2*0<poi_tan_left_temp[0]<poi_patch_output[1,0]-self.h/2*0 and poi_patch_output[0,1]+self.h/2*0<poi_tan_left_temp[1]<poi_patch_output[2,1]-self.h/2*0):
                        continue
                    poi_tan_left = poi_tan_left_temp
                    order_tan_left = order_tan_left_temp
                    poi_base_left = poi_base
                    LeftTangentHandled = True

                if not RightTangentHandled and order_tan_right_temp>=0:
                    # 判断切点是否在output内，若是，则换点
                    if (poi_patch_output[0,0]+self.h/2*0<poi_tan_right_temp[0]<poi_patch_output[1,0]-self.h/2*0 and poi_patch_output[0,1]+self.h/2*0<poi_tan_right_temp[1]<poi_patch_output[2,1]-self.h/2*0):
                        continue
                    poi_tan_right = poi_tan_right_temp
                    order_tan_right = order_tan_right_temp
                    poi_base_right = poi_base
                    RightTangentHandled = True

            if order_tan_left == -1 or order_tan_right == -1:
                p_l_patch_radial_temp = [p_l_patch[ShapeOrder]]
                p_l_boundary_patch_radial_temp = [p_l_patch[ShapeOrder]]
            else:
                if order_tan_left<=order_tan_right:
                    p_l_patch_boundary_radial = list(p_l_patch[ShapeOrder][order_tan_left:order_tan_right+1,:])
                else:
                    p_l_patch_boundary_radial = list(p_l_patch[ShapeOrder][order_tan_left:, :])
                    p_l_patch_boundary_radial.extend(p_l_patch[ShapeOrder][1:order_tan_right+1, :])

                poi_intersect = []
                line_intersect = []

                s_poi = poi_base_left
                e_poi = poi_tan_left + 1000 * (poi_tan_left-s_poi)
                poi, line = self.isPatchIntersectLine(poi_patch, s_poi, e_poi)
                poi_intersect.append(poi)
                line_intersect.append(line)
                if not (poi[0] == p_l_patch_boundary_radial[0][0] and poi[1] == p_l_patch_boundary_radial[0][1]):
                    p_l_patch_boundary_radial.insert(0, poi)

                s_poi = poi_base_right
                e_poi = poi_tan_right + 1000 * (poi_tan_right - s_poi)
                poi, line = self.isPatchIntersectLine(poi_patch, s_poi, e_poi)
                poi_intersect.append(poi)
                line_intersect.append(line)
                if not (poi[0] == p_l_patch_boundary_radial[-1][0] and poi[1] == p_l_patch_boundary_radial[-1][1]):
                    p_l_patch_boundary_radial.append(poi)

                poi_intersect = np.array(poi_intersect)

                p_l_patch_radial_temp, p_l_boundary_patch_radial_temp = self.Get_p_l_patch(poi_patch,poi_intersect,line_intersect,[p_l_patch_boundary_radial])

            p_l_patch_radial.extend(np.array(p_l_patch_radial_temp))
            p_l_boundary_patch_radial.extend(np.array(p_l_boundary_patch_radial_temp))

        return p_l_patch_radial, p_l_boundary_patch_radial, poi_tan_left, poi_tan_right

    def GenerateFluidPoint(self, poi_patch_output, IsInPoly_output):
        FirstPoint = False
        assert max(IsInPoly_output)>0
        if min(IsInPoly_output) == 1:
            poi_base = [poi_patch_output[0,0] + (self.PatchLength_output//2)*self.h+self.h/2,poi_patch_output[0,1] + (self.PatchLength_output//2)*self.h+self.h/2]
            yield poi_base


        for a in range(self.PatchLength_output//2):
            for b in range(self.PatchLength_output//2):

                aa = a
                bb = b
                if IsInPoly_output[aa + bb * self.PatchLength_output] > 0:
                    poi_base = [poi_patch_output[0, 0] + aa * self.h + self.h / 2,
                                poi_patch_output[0, 1] + bb * self.h + self.h / 2]
                    yield poi_base
                aa = a
                bb = self.PatchLength_output - b - 1

                if IsInPoly_output[aa + bb * self.PatchLength_output] > 0:
                    poi_base = [poi_patch_output[0, 0] + aa * self.h + self.h / 2,
                                poi_patch_output[0, 1] + bb * self.h + self.h / 2]
                    # print('bbb1')
                    yield poi_base
                aa = self.PatchLength_output - a - 1
                bb = b

                if IsInPoly_output[aa + bb * self.PatchLength_output] > 0:
                    poi_base = [poi_patch_output[0, 0] + aa * self.h + self.h / 2,
                                poi_patch_output[0, 1] + bb * self.h + self.h / 2]
                    # print('bbb1')
                    yield poi_base
                aa = self.PatchLength_output - a - 1
                bb = self.PatchLength_output - b - 1

                if IsInPoly_output[aa + bb * self.PatchLength_output] > 0:
                    poi_base = [poi_patch_output[0, 0] + aa * self.h + self.h / 2,
                                poi_patch_output[0, 1] + bb * self.h + self.h / 2]
                    # print('bbb1')
                    yield poi_base

        yield False

    def GetTangent(self, poi, p_l):
        # 输入：多边形外一点，多边形顶点坐标
        # 输出：两个切点
        tan_left = p_l[0, :]
        tan_right = p_l[0, :]
        order_left = -1
        order_right = -1
        flag1 = self.isLeft(p_l[0, :], p_l[1, :], poi)
        for i in range(1, len(p_l[:, 0]) - 1):
            flag2 = self.isLeft(p_l[i, :], p_l[i + 1, :], poi)

            if flag1 <= 0 and flag2 > 0:
                tan_right = p_l[i, :]
                order_right = i
            elif flag1 > 0 and flag2 <= 0:
                '''print('flag1=', flag1)
                print('flag2=', flag2)
                print('result2=', self.isLeft(poi, p_l[i, :], tan_left))
                print('tan_left =', p_l[i, :])'''
                tan_left = p_l[i, :]
                order_left = i
            flag1 = flag2
        flag2 = self.isLeft(p_l[-1, :], p_l[1, :], poi)
        if flag1 <= 0 and flag2 > 0:
            '''print('flag1=', flag1)
            print('flag2=', flag2)
            print('result1=', self.isLeft(poi, p_l[-1, :], tan_right))'''
            tan_right = p_l[-1, :]
            order_right = 0
        elif flag1 > 0 and flag2 <= 0:
            '''print('flag1=', flag1)
            print('flag2=', flag2)
            print('result2=', self.isLeft(poi, p_l[-1, :], tan_left))'''
            tan_left = p_l[-1, :]
            order_left = 0

        return tan_left, tan_right, order_left, order_right

    def isLeft(self, s_poi, e_poi, poi):
        # 判断poi在线段的左侧还是右侧
        # >0左侧，=0在线段上，<0在右侧
        return (e_poi[0]-s_poi[0])*(poi[1]-s_poi[1])-(poi[0]-s_poi[0])*(e_poi[1]-s_poi[1])




























