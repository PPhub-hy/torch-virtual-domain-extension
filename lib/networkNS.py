mport torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from lib.utils import generte_legendre_filters_2D, generte_legendre_filters_1D, generte_boundary_filters_2D
from lib.utils import spatial_gradient

class NetNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, In_length, cheb = False):
        super(NetNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 6
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_r, self.filter_d, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 2 * In_length
        out_channel = 2

        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        input = input / 5
        recept = self.recept
        r = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        input = F.pad(input, (left, right, left, right), "circular")

        x = self.first_conv(input)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k)
            Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        x = x * 5

        return x[:,:,:-r,:-r]

class NetComNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, Params = None, cheb = False):
        super(NetComNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.norm_factors = Params['norm_factors'] # [0.33, 0.33, 1, 1]
        self.if_ln = Params['if_ln']
        #self.IfFactor = Params['IfFactor'] # True
        #self.norm_max = Params['norm_max'] # [3.0856, 3.0856, 1.7608, 1.2852]
        #self.norm_min = Params['norm_min'] # [0, 0, 0.6272, 0.8340]
        print('norm_factors:{} (u,v,rho,T)'.format(self.norm_factors))
        print('if_ln:{}',format(self.if_ln))
        self.n = Params['n'] #12
        self.m = Params['m'] #6
        self.k = Params['k'] #2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_d, self.filter_r, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 4
        out_channel = 4

        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv_kernel_size = 0
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=2*self.first_conv_kernel_size+1, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        # Normalize the initial weights
        self.init_weight = Params['init_weight']

        self.first_conv.weight = nn.Parameter(self.first_conv.weight * self.init_weight[0])
        for i in range(num_blocks):
            self.linearModes[i].weight = nn.Parameter(self.linearModes[i].weight * self.init_weight[1])
            self.convs[i][0].weight = nn.Parameter(self.convs[i][0].weight * self.init_weight[2])
            self.convs[i][2].weight = nn.Parameter(self.convs[i][2].weight * self.init_weight[2])
            self.convs[i][4].weight = nn.Parameter(self.convs[i][4].weight * self.init_weight[2])
            #self.convs[i][6].weight = nn.Parameter(self.convs[i][6].weight * self.init_weight[2])
            #self.convs[i][8].weight = nn.Parameter(self.convs[i][8].weight * self.init_weight[2])
            self.convs[i][-1].weight = nn.Parameter(self.convs[i][-1].weight * self.init_weight[3])
        self.conv11.weight = nn.Parameter(self.conv11.weight * self.init_weight[4])
        self.convout.weight = nn.Parameter(self.convout.weight * self.init_weight[5])

        print('init_weight=:sqrt{} (first_conv, LinearModes, convs1, convs2, conv11, convout)'.format(np.array(self.init_weight)**2))


        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + self.first_conv_kernel_size
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        input_ln = input.clone()
        '''if self.if_ln:
            input_ln[:, 2:4, :, :] = torch.log(input[:, 2:4, :, :]).clone()'''

        for i in range(input_ln.shape[1]):
            input_ln[:, i, :, :] = input_ln[:, i, :, :] * self.norm_factors[i]
        #input = F.interpolate(input, size=(input.shape[-1]*2,input.shape[-1]*2), mode='bicubic')

        recept = self.recept
        r = self.get_padding_R(input_ln.shape[-1] + 2 * recept - 2 * self.first_conv_kernel_size)
        left = recept
        right = recept + r
        #input_ln = F.pad(input_ln, (left, right, left, right), "circular")

        x = self.first_conv(input_ln)
        #x = F.gelu(x)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 4
            assert rc <= RR

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            #Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, self.modes * c, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            #Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2 * 1

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k)
            if rc == RR:
                Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:, :, RR:-RR, RR:-RR] + linear_x[:, :, :, :]
            else:
                Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        for i in range(input.shape[1]):
            x[:, i, :, :] = x[:, i, :, :] / self.norm_factors[i]

        #x = x[:, :, :-r, :-r]
        xx = x.clone()
        '''if self.if_ln:
            xx[:, 2:4, :, :] = torch.pow(x[:, 2:4, :, :],2).clone()'''
        #x = F.interpolate(x, size=(x.shape[-1] // 2, x.shape[-1] // 2), mode='bicubic')
        return xx

class NetWave_InN_legendre(nn.Module):

    def __init__(self, num_blocks, cheb = False):
        super(NetWave_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 4
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_r, self.filter_d, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []
        first_channel = 2
        out_channel = 1
        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2
        if l_input % n == 0:
            return 0
        else:
            return n - l_input % n

    def forward(self, input):
        input[:, 1:2, :, :] = input[:, 1:2, :, :] / 10
        p = input[:, 0:1, :, :]
        dp = input[:, 1:2, :, :]

        recept = self.recept
        r = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        x = F.pad(input, (left, right, left, right), "circular")

        x = self.first_conv(x)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l = x.shape[-1]
            x = x.reshape(b * c, 1, l, l)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll = Legendre_x.shape[-1]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll, ll)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll, ll)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll, ll)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll, ll)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k) #shape: [b * c, self.modes, l, l]
            Legendre_x = Legendre_x.reshape(b, c, l, l)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        x = (dp + x[:,:,:-r,: -r]) * 10
        out = torch.cat((p + 0.05 * x, x), 1)

        return out

'''
The network used for Burgers2D problem is identical with the 'NetNS_InN_legendre'
'''
class NetBurgers1D_InN_legendre(nn.Module):

    def __init__(self, num_blocks, In_length, cheb = False):
        super(NetBurgersn_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False

        self.filter_d = None
        self.filter_r = None

        self.n = 12
        self.m = 6
        self.k = 2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_r, self.filter_d, self.l_filters = generte_legendre_filters_1D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []
        first_channel = 1 * In_length
        out_channel = 1
        channel = 20
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv1d(first_channel, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv1d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv1d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv1d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv1d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        recept = self.recept
        r = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        input = F.pad(input, (left, right), "circular")

        x = self.first_conv(input)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l = x.shape[-1]
            x = x.reshape(b * c, 1, l)
            Legendre_x = F.conv1d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll = Legendre_x.shape[-1]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose1d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k) #shape: [b * c, self.modes, l, l]
            Legendre_x = Legendre_x.reshape(b, c, l)[:,:,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        return x[:,:,:-r]

class NetComNSBoundary_legendre(nn.Module):

    def __init__(self, num_blocks, normal_N, cheb = False):
        super(NetComNSBoundary_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.normal_N = normal_N
        self.normal_m = 10
        self.n = 12
        self.m = 6
        self.k = 2
        print('tangential params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        print('normal params: normal_N:{} normal_m:{}'.format(self.normal_N, self.normal_m))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_r, self.filter_d, self.l_filters = generte_boundary_filters_2D(file_path,
                                                                                   n=self.n, m=self.m,
                                                                                   normal_N=self.normal_N,
                                                                                   normal_m=self.normal_m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.curvature_convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 4
        out_channel = 4

        channel = 40
        channel_curvature = 64
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=(3,1), bias=with_bias)

        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.curvature_convs.append(
                nn.Sequential(
                    nn.Conv1d(1, channel_curvature, kernel_size=self.n, stride=self.n // self.k, bias=with_bias),
                    nn.GELU(),
                    nn.Conv1d(channel_curvature, self.modes, kernel_size=1, bias=with_bias),
                            )
            )
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=(3,1), bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=(3,1), bias=with_bias),
                            )
            )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.curvature_convs = nn.ModuleList(self.curvature_convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        '''
        recept = self.recept
        r = self.get_padding_R(input.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        input = F.pad(input, (left, right, left, right), "circular")
        '''
        x = input[0]
        curvature = input[1]

        '''
        dx_normal = 1/128
        neg_r = torch.arange(0, x.shape[-1] * dx_normal, step=dx_normal).reshape(1, x.shape[-1]).cuda()
        transfer_factors = 1 - torch.matmul(curvature.unsqueeze(-1), neg_r)
        transfer_factors = 1/transfer_factors
        ut = x[:,1:2,:,:]
        ut = ut.mul(transfer_factors)
        x[:, 1:2, :, :] = ut'''


        x = self.first_conv(x)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d, stride=(self.n // self.k,1))  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]


            '''Legendre_curvature = self.curvature_convs[idx](curvature).unsqueeze(-1)
            if not idx == 0:
                Legendre_curvature = Legendre_curvature[:,:,idx:-idx,:]
            assert ll1 == 1
            assert ll2 == Legendre_curvature.shape[-2]
            Legendre_curvature = Legendre_curvature.repeat(c,1,1,1)
            #curvature = curvature.repeat(c // 4,1,1,1)
            #Legendre_x_partA = Legendre_x[:b * c // 4]
            #Legendre_x_partB = Legendre_x[b * c // 4:]
            #Legendre_x_partA = Legendre_x_partA.mul(curvature)
            #Legendre_x = torch.cat([Legendre_x_partA, Legendre_x_partB],dim=0)
            Legendre_x = Legendre_x.mul(Legendre_curvature)'''


            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=(self.n // self.k,1))
            Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,:] + linear_x[:,:,RR-rc:-RR+rc,:]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        '''
        ut = x[:, 1:2, :, :]
        ut = ut.mul(1 / transfer_factors[:,:,self.recept:-self.recept,:])
        x[:, 1:2, :, :] = ut'''

        return x[:,:,:,:]

class RefinedNetComNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, Params = None, cheb = False):
        super(RefinedNetComNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False#True#

        self.filter_d = None
        self.filter_r = None

        self.norm_factors = Params['norm_factors']
        print('norm_factors:{} (u,v,rho,T)'.format(self.norm_factors))
        self.n = Params['n'] #12
        self.m = Params['m'] #6
        self.k = Params['k'] #2
        print('legendre params: n:{} m:{} k:{}'.format(self.n,self.m,self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*"*5, 'using Chebyshev filters', "*"*5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*"*5,'using Legendre filters',"*"*5)
        self.filter_r, self.filter_d, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 4
        out_channel = 4

        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=3, bias=with_bias)
        self.first_conv_grad = nn.Conv2d(first_channel * 2, channel, kernel_size=3, bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                            )
                        )

        self.conv11 = nn.Conv2d(channel, 128,  kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(128, out_channel, kernel_size=1, bias=with_bias)

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + 1
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % 2 == 0
        n = n // 2

        return n - l_input % n

    def forward(self, input):
        x = input
        for i in range(input.shape[1]):
            x[:, i, :, :] = x[:, i, :, :] * self.norm_factors[i]
        #input = F.interpolate(input, size=(input.shape[-1]*2,input.shape[-1]*2), mode='bicubic')

        recept = self.recept
        r = self.get_padding_R(x.shape[-1] + 2 * recept - 2)
        left = recept
        right = recept + r
        x = F.pad(x, (left, right, left, right), "circular")

        x_grad = spatial_gradient(x)
        x = self.first_conv(x)
        x_grad = self.first_conv_grad(x_grad)
        #x = x * x_grad

        b = x.shape[0]
        c = x.shape[1]
        for idx in range(self.num_blocks):
            RR = self.RR

            linear_x = self.convs[idx](x)
            rc = 2

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x = x.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x, self.filter_d / self.k, stride=self.n // self.k)  #shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.permute(0, 2, 1, 3, 4)
            Legendre_x = Legendre_x.reshape(b, c * self.modes, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)
            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.permute(0, 2, 1, 3, 4)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2*2

            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r / self.k, stride=self.n // self.k)
            Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:,:,RR:-RR,RR:-RR] + linear_x[:,:,RR-rc:-RR+rc,RR-rc:-RR+rc]

            x = F.gelu(Legendre_x)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        for i in range(input.shape[1]):
            x[:, i, :, :] = x[:, i, :, :] / self.norm_factors[i]

        x = x[:, :, :-r, :-r]
        #x = F.interpolate(x, size=(x.shape[-1] // 2, x.shape[-1] // 2), mode='bicubic')
        return x + input
