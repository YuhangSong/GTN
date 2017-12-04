import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian
import numpy as np
from arguments import gtn_M, gtn_N, hierarchical, parameter_noise_rate, both_side_tower, multi_gpu, gpus, loss_fisher_sensitivity_per_m
from arguments import log_fisher_sensitivity_per_m
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def to_data_parallel(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m = nn.DataParallel(m, device_ids=gpus)

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x, _ = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x, conv_list = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy, conv_list

    def evaluate_states_value_fisher(self, inputs):
        value, _, _ = self(inputs)
        value = value.log()
        return value

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()

        self.final_flatten_size = []
        for m in range(gtn_M):
            if hierarchical == 0:
                depth = gtn_N
            elif hierarchical == 1:
                depth = gtn_N + m
            ''' current m-level final feature size'''
            final_number_feature = 32*(2**(depth-1))
            final_size = 128/(2**(depth))
            self.final_flatten_size += [int(final_number_feature * (final_size**2))]
        self.final_flatten_size += [0]*10

        if hierarchical == 1:

            if gtn_M >= 1:

                m = 0 ###############

                if gtn_N >= 1:
                    # 4 128 128
                    self.conv00 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv01 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv02 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv03 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv04 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv05 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_0 = nn.Linear(temp, 512)

            if gtn_M >= 2:

                m = 1 ###############

                if gtn_N >= 1:
                    # 32 64 64
                    self.conv10 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 64 32 32
                    self.conv11 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 128 16 16
                    self.conv12 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 256 8 8
                    self.conv13 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 512 4 4
                    self.conv14 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_1 = nn.Linear(temp, 512)

            if gtn_M >= 3:

                m = 2 ###############

                if gtn_N >= 1:
                    # 64 32 32
                    self.conv20 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 128 16 16
                    self.conv21 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 256 8 8
                    self.conv22 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 512 4 4
                    self.conv23 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_2 = nn.Linear(temp, 512)

            if gtn_M >= 4:

                m = 3 ###############

                if gtn_N >= 1:
                    # 128 16 16
                    self.conv30 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 256 8 8
                    self.conv31 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 512 4 4
                    self.conv32 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_3 = nn.Linear(temp, 512)

            if gtn_M >= 5:

                m = 4 ###############

                if gtn_N >= 1:
                    # 256 8 8
                    self.conv40 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 512 4 4
                    self.conv41 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_4 = nn.Linear(temp, 512)

            if gtn_M >= 6:

                m = 5 ###############

                if gtn_N >= 1:
                    # 512 4 4
                    self.conv50 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

                if both_side_tower == 1:
                    temp = self.final_flatten_size[m]+512*(int(np.clip((gtn_M-m-1), a_min=0, a_max=1)))
                    if temp > 0:
                        self.linear_cat_5 = nn.Linear(temp, 512)

        elif hierarchical == 0:
            if gtn_M >= 1:
                m = 0 ###############
                if gtn_N>=1:
                    # 4 128 128
                    self.conv00 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv01 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv02 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv03 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv04 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv05 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

            if gtn_M >= 2:
                m = 1 ###############
                if gtn_N >= 1:
                    # 4 128 128
                    self.conv10 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv11 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv12 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv13 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv14 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv15 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

            if gtn_M >= 3:
                m = 2 ###############
                if gtn_N >= 1:
                    # 4 128 128
                    self.conv20 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv21 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv22 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv23 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv24 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv25 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

            if gtn_M >= 4:
                m = 3 ###############
                if gtn_N >= 1:    
                    # 4 128 128
                    self.conv30 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv31 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv32 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv33 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv34 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv35 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

            if gtn_M >= 5:
                m = 4 ###############
                if gtn_N >= 1:
                    # 4 128 128
                    self.conv40 = nn.Conv2d(
                        in_channels=num_inputs,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 2:
                    # 32 64 64
                    self.conv41 = nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 3:
                    # 64 32 32
                    self.conv42 = nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 4:
                    # 128 16 16
                    self.conv43 = nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 5:
                    # 256 8 8
                    self.conv44 = nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                if gtn_N >= 6:
                    # 512 4 4
                    self.conv45 = nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
                    # 1024 2 2

        if both_side_tower == 0:
            self.concatenation_layer_size = sum(self.final_flatten_size)
            self.concatenation_layer = nn.Linear(self.concatenation_layer_size, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()
        if multi_gpu == 1:
            self.apply(to_data_parallel)

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')

        def reset_linear_conv_parameters(x):
            x.weight.data.mul_(relu_gain)

        import inspect
        def retrieve_name(var):
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            return [var_name for var_name, var_val in callers_local_vars if var_val is var]

        if True:
            try:
                reset_linear_conv_parameters(self.conv00)
                print('conv00')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv01)
                print('conv01')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv02)
                print('conv02')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv03)
                print('conv03')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv04)
                print('conv04')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv05)
                print('conv05')
            except Exception as e:
                pass

            try:
                reset_linear_conv_parameters(self.conv10)
                print('conv10')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv11)
                print('conv11')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv12)
                print('conv12')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv13)
                print('conv13')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv14)
                print('conv14')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv15)
                print('conv15')
            except Exception as e:
                pass

            try:
                reset_linear_conv_parameters(self.conv20)
                print('conv20')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv21)
                print('conv21')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv22)
                print('conv22')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv23)
                print('conv23')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv24)
                print('conv24')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv25)
                print('conv25')
            except Exception as e:
                pass

            try:
                reset_linear_conv_parameters(self.conv30)
                print('conv30')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv31)
                print('conv31')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv32)
                print('conv32')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv33)
                print('conv33')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv34)
                print('conv34')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv35)
                print('conv35')
            except Exception as e:
                pass

            try:
                reset_linear_conv_parameters(self.conv40)
                print('conv40')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv41)
                print('conv41')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv42)
                print('conv42')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv43)
                print('conv43')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv44)
                print('conv44')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv45)
                print('conv45')
            except Exception as e:
                pass

            try:
                reset_linear_conv_parameters(self.conv50)
                print('conv50')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv51)
                print('conv51')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv52)
                print('conv52')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv53)
                print('conv53')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv54)
                print('conv54')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.conv55)
                print('conv55')
            except Exception as e:
                pass

        if True:
            try:
                reset_linear_conv_parameters(self.concatenation_layer)
                print('concatenation_layer')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_0)
                print('linear_cat_0')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_1)
                print('linear_cat_1')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_2)
                print('linear_cat_2')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_3)
                print('linear_cat_3')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_4)
                print('linear_cat_4')
            except Exception as e:
                pass
            try:
                reset_linear_conv_parameters(self.linear_cat_5)
                print('linear_cat_5')
            except Exception as e:
                pass
        

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        
        x0 = inputs / 255.0

        conv_list = []

        if gtn_M >= 1:

            if hierarchical==0:
                x1 = x0

            if gtn_N >= 1:
                x0 = self.conv00(x0)
                x0 = F.relu(x0)

            if hierarchical==1:
                x1 = x0

            conv_list += [x0]

            if gtn_N >= 2:
                x0 = self.conv01(x0)
                x0 = F.relu(x0)

            if gtn_N >= 3:
                x0 = self.conv02(x0)
                x0 = F.relu(x0)

            if gtn_N >= 4:
                x0 = self.conv03(x0)
                x0 = F.relu(x0)

            if gtn_N >= 5:
                x0 = self.conv04(x0)
                x0 = F.relu(x0)

            if gtn_N >= 6:
                x0 = self.conv05(x0)
                x0 = F.relu(x0)

            x0 = x0.view(-1, x0.size()[1]*x0.size()[2]*x0.size()[3])

        if gtn_M >= 2:

            if hierarchical==0:
                x2 = x1

            if gtn_N >= 1:
                x1 = self.conv10(x1)
                x1 = F.relu(x1)

            if hierarchical==1:
                x2 = x1

            conv_list += [x1]

            if gtn_N >= 2:
                x1 = self.conv11(x1)
                x1 = F.relu(x1)

            if gtn_N >= 3:
                x1 = self.conv12(x1)
                x1 = F.relu(x1)

            if gtn_N >= 4:
                x1 = self.conv13(x1)
                x1 = F.relu(x1)

            if gtn_N >= 5:
                x1 = self.conv14(x1)
                x1 = F.relu(x1)

            x1 = x1.view(-1, x1.size()[1]*x1.size()[2]*x1.size()[3])

        if gtn_M >= 3:

            if hierarchical==0:
                x3 = x2

            if gtn_N >= 1:
                x2 = self.conv20(x2)
                x2 = F.relu(x2)

            if hierarchical==1:
                x3 = x2

            conv_list += [x2]

            if gtn_N >= 2:
                x2 = self.conv21(x2)
                x2 = F.relu(x2)

            if gtn_N >= 3:
                x2 = self.conv22(x2)
                x2 = F.relu(x2)

            if gtn_N >= 4:
                x2 = self.conv23(x2)
                x2 = F.relu(x2)

            x2 = x2.view(-1, x2.size()[1]*x2.size()[2]*x2.size()[3])

        if gtn_M >= 4:

            if hierarchical==0:
                x4 = x3

            if gtn_N >= 1:
                x3 = self.conv30(x3)
                x3 = F.relu(x3)

            if hierarchical==1:
                x4 = x3

            conv_list += [x3]

            if gtn_N >= 2:
                x3 = self.conv31(x3)
                x3 = F.relu(x3)

            if gtn_N >= 3:
                x3 = self.conv32(x3)
                x3 = F.relu(x3)

            x3 = x3.view(-1, x3.size()[1]*x3.size()[2]*x3.size()[3])

        if gtn_M >= 5:

            if hierarchical==0:
                x5 = x4

            if gtn_N >= 1:
                x4 = self.conv40(x4)
                x4 = F.relu(x4)

            if hierarchical==1:
                x5 = x4

            if gtn_N >= 2:
                x4 = self.conv41(x4)
                x4 = F.relu(x4)

            conv_list += [x4]

            x4 = x4.view(-1, x4.size()[1]*x4.size()[2]*x4.size()[3])

        if gtn_M >= 6:

            if hierarchical==0:
                x6 = x5

            if gtn_N >= 1:
                x5 = self.conv50(x5)
                x5 = F.relu(x5)

            if hierarchical==1:
                x6 = x5

            conv_list += [x5]

            x5 = x5.view(-1, x5.size()[1]*x5.size()[2]*x5.size()[3])

        if both_side_tower == 1:

            if gtn_M >= 6:

                if gtn_M >= 7:
                    x5 = self.linear_cat_5(torch.cat([x5,x6],1))
                else:
                    x5 = self.linear_cat_5(x5)

                x5 = F.relu(x5)

            if gtn_M >= 5:

                if gtn_M >= 6:
                    x4 = self.linear_cat_4(torch.cat([x4,x5],1))
                else:
                    x4 = self.linear_cat_4(x4)

                x4 = F.relu(x4)

            if gtn_M >= 4:

                if gtn_M >= 5:
                    x3 = self.linear_cat_3(torch.cat([x3,x4],1))
                else:
                    x3 = self.linear_cat_3(x3)

                x3 = F.relu(x3)

            if gtn_M >= 3:

                if gtn_M >= 4:
                    x2 = self.linear_cat_2(torch.cat([x2,x3],1))
                else:
                    x2 = self.linear_cat_2(x2)

                x2 = F.relu(x2)

            if gtn_M >= 2:

                if gtn_M >= 3:
                    x1 = self.linear_cat_1(torch.cat([x1,x2],1))
                else:
                    x1 = self.linear_cat_1(x1)

                x1 = F.relu(x1)

            if gtn_M >= 1:

                if gtn_M >= 2:
                    x0 = self.linear_cat_0(torch.cat([x0,x1],1))
                else:
                    x0 = self.linear_cat_0(x0)

                x0 = F.relu(x0)

        if gtn_M == 1:
            if both_side_tower == 1:
                x = x0

            else:
                x = self.concatenation_layer(x0)
                x = F.relu(x)

        else:
            if both_side_tower == 1:
                x = x0

            else:
                if gtn_M == 2:
                    x = [x0,x1]
                elif gtn_M == 3:
                    x = [x0,x1,x2]
                elif gtn_M == 4:
                    x = [x0,x1,x2,x3]
                elif gtn_M == 5:
                    x = [x0,x1,x2,x3,x4]
                else:
                    raise Exception('Not support')
                x = self.concatenation_layer(torch.cat(x,1))
                x = F.relu(x)

        return self.critic_linear(x), x, None #conv_list

    def parameter_noise(self):
        for p in self.parameters():
            p.data = torch.normal(
                means=p.data,
                std=p.data.abs()*parameter_noise_rate,
                )

    ########################### for AFS ###############################

    def get_gradient_reward_one_m(self, action_log_probs, conv):

        def get_grad_norm(inputs,outputs):

            gradients = autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=torch.ones(outputs.size()).cuda(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.contiguous()
            gradients_fl = gradients.view(gradients.size()[0],-1)
            gradients_norm = gradients_fl.norm(2, dim=1) / ((gradients_fl.size()[1])**0.5)

            return gradients_norm

        gradients_norm_noise = get_grad_norm(conv,action_log_probs)

        gradients_reward = (gradients_norm_noise+1.0).log().mean()

        return gradients_reward

    def get_afs_per_m(self, action_log_probs, conv_list):
        '''Average Fisher Sensitivity (AFS)'''
        
        afs_per_m = []

        if log_fisher_sensitivity_per_m == 1:
            for m in range(gtn_M):
                afs_per_m += [self.get_gradient_reward_one_m(action_log_probs, conv_list[m])]

        loss_afs = None
        if loss_fisher_sensitivity_per_m==1:
            for m in range(len(afs_per_m)):
                if afs_per_m[m].data.cpu().numpy()[0]==0.0:
                    continue
                else:
                    temp = afs_per_m[m] * (m) * 0.1
                    if loss_afs is not None:
                        loss_afs += temp
                    else:
                        loss_afs = temp.clone()

        if len(afs_per_m)>0:
            for m in range(len(afs_per_m)):
                afs_per_m[m] = afs_per_m[m].data.cpu().numpy()[0]

        return afs_per_m, loss_afs

    ########################### for EWC ###############################

    def compute_fisher(self, states, num_samples=200, plot_diffs=False, disp_freq=10):

        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        # self.F_accum = []
        # for p in list(self.parameters()):
        #     self.F_accum += [p.data.cpu().numpy().fill(0.0)]

        # sampling a random class from softmax
        # probs = tf.nn.softmax(self.y)
        # class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        # if(plot_diffs):
        #     # track differences in mean Fisher info
        #     F_prev = deepcopy(self.F_accum)
        #     mean_diffs = np.zeros(0)

        

        def get_fisher_one(state):
            value = self.evaluate_states_value_fisher(
                inputs = autograd.Variable(state),
            ).sum()
            value.backward()

            parameters_have_grad_index = []
            for p in self.parameters():
                if p.grad is not None:
                    parameters_have_grad_index += [True]
                else:
                    parameters_have_grad_index += [False]

            F_one = []
            for p, have_grad in zip(self.parameters(), parameters_have_grad_index):
                if have_grad:
                    F_one += [p.grad.data.pow(2)]
                else:
                    pass

            return F_one, parameters_have_grad_index

        for b in range(states.size()[0]):
            state = states[b:(b+1)]
            F_one, self.parameters_have_grad_index = get_fisher_one(state)
            try:
                for ii in range(len(F_one)):
                    self.F_accum[ii] = self.F_accum[ii] + F_one[ii]
            except Exception as e:
                self.F_accum = F_one

        for ii in range(len(self.F_accum)):
            self.F_accum[ii] = self.F_accum[ii] / states.size()[0]

        # for i in range(num_samples):
        #     # select random input image
        #     im_ind = np.random.randint(imgset.shape[0])
        #     # compute first-order derivatives
        #     ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
        #     # square the derivatives and add to total
        #     for v in range(len(self.F_accum)):
        #         self.F_accum[v] += np.square(ders[v])
        #     # if(plot_diffs):
        #     #     if i % disp_freq == 0 and i > 0:
        #     #         # recording mean diffs of F
        #     #         F_diff = 0
        #     #         for v in range(len(self.F_accum)):
        #     #             F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
        #     #         mean_diff = np.mean(F_diff)
        #     #         mean_diffs = np.append(mean_diffs, mean_diff)
        #     #         for v in range(len(self.F_accum)):
        #     #             F_prev[v] = self.F_accum[v]/(i+1)
        #     #         plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
        #     #         plt.xlabel("Number of samples")
        #     #         plt.ylabel("Mean absolute Fisher difference")
        #     #         display.display(plt.gcf())
        #     #         display.clear_output(wait=True)

        # # divide totals by number of samples
        # for v in range(len(self.F_accum)):
        #     self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for p, have_grad in zip(self.parameters(), self.parameters_have_grad_index):
            if have_grad:
                self.star_vars += [p.data.clone()]
            else:
                pass

    def get_ewc_loss(self, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        try:
            temp = self.star_vars[0]
            temp = self.F_accum[0]
        except Exception as e:
            return None

        ii = 0
        for p, have_grad in zip(self.parameters(), self.parameters_have_grad_index):
            
            if have_grad:

                temp = (p - Variable(self.star_vars[ii])).pow(2)
                loss = (lam/2) * (torch.mul(Variable(self.F_accum[ii]),temp)).sum()
                
                try:
                    ewc_loss += loss
                    
                except Exception as e:
                    ewc_loss = loss

                ii += 1

        return ewc_loss

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x
