import torch
import torch.nn as nn
import torch.nn.functional as F
from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian

from arguments import gtn_M, gtn_N, hierarchical, parameter_noise_rate

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()

        if hierarchical == 1:

            ############ m = 0 ###############

            # 4 128 128
            self.conv00 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv01 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv02 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv03 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv04 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

            ############ m = 1 ###############

            # 32 64 64
            self.conv10 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv11 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv12 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv13 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4
            self.conv23 = nn.Conv2d(
                in_channels=512,
                out_channels=2024,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 1024 2 2

            ############ m = 2 ###############

            # 64 32 32
            self.conv20 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv21 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv22 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4
            self.conv23 = nn.Conv2d(
                in_channels=512,
                out_channels=2024,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 1024 2 2

            ############ m = 3 ###############

            # 128 16 16
            self.conv30 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv31 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4
            self.conv32 = nn.Conv2d(
                in_channels=512,
                out_channels=2024,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 1024 2 2

            ############ m = 4 ###############

            # 256 8 8
            self.conv40 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4
            self.conv41 = nn.Conv2d(
                in_channels=512,
                out_channels=2024,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 1024 2 2

            ############ m = 5 ###############

            # 512 4 4
            self.conv50 = nn.Conv2d(
                in_channels=512,
                out_channels=2024,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 1024 2 2

        elif hierarchical == 0:

            ############ m = 0 ###############

            # 4 128 128
            self.conv00 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv01 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv02 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv03 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv04 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

            ############ m = 1 ###############

            # 4 128 128
            self.conv10 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv11 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv12 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv13 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv14 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

            ############ m = 2 ###############

            # 4 128 128
            self.conv20 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv21 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv22 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv23 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv24 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

            ############ m = 3 ###############

            # 4 128 128
            self.conv30 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv31 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv32 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv33 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv34 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

            ############ m = 4 ###############

            # 4 128 128
            self.conv40 = nn.Conv2d(
                in_channels=num_inputs,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 32 64 64
            self.conv41 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 64 32 32
            self.conv42 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 128 16 16
            self.conv43 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 256 8 8
            self.conv44 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                )
            # 512 4 4

        self.concatenation_layer_size = 0
        for m in range(gtn_M):
            if hierarchical == 0:
                depth = gtn_N
            elif hierarchical == 1:
                depth = gtn_N + m
            final_number_feature = 32*(2**(depth-1))
            final_size = 128/(2**(depth))
            self.concatenation_layer_size += final_number_feature * (final_size**2)
        self.concatenation_layer_size = int(self.concatenation_layer_size)

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

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')

        self.conv00.weight.data.mul_(relu_gain)
        self.conv01.weight.data.mul_(relu_gain)
        self.conv02.weight.data.mul_(relu_gain)

        self.concatenation_layer.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        
        x0 = inputs / 255.0

        if gtn_M >= 1:

            if hierarchical==0:
                x1 = x0

            if gtn_N >= 1:
                x0 = self.conv00(x0)
                x0 = F.relu(x0)

            if hierarchical==1:
                x1 = x0

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

            x0 = x0.view(-1, x0.size()[1]*x0.size()[2]*x0.size()[3])

        if gtn_M >= 2:

            if hierarchical==0:
                x2 = x1

            if gtn_N >= 1:
                x1 = self.conv10(x1)
                x1 = F.relu(x1)

            if hierarchical==1:
                x2 = x1

            if gtn_N >= 2:
                x1 = self.conv11(x1)
                x1 = F.relu(x1)

            if gtn_N >= 3:
                x1 = self.conv12(x1)
                x1 = F.relu(x1)

            if gtn_N >= 4:
                x1 = self.conv13(x1)
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

            if gtn_N >= 2:
                x2 = self.conv21(x2)
                x2 = F.relu(x2)

            if gtn_N >= 3:
                x2 = self.conv22(x2)
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

            if gtn_N >= 2:
                x3 = self.conv31(x3)
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

            x4 = x4.view(-1, x4.size()[1]*x4.size()[2]*x4.size()[3])

        if gtn_M == 1:
            x = self.concatenation_layer(x0)
        else:
            if gtn_M == 2:
                x = [x0,x1]
            elif gtn_M == 3:
                x = [x0,x1,x2]
            x = self.concatenation_layer(torch.cat(x,1))

        x = F.relu(x)

        return self.critic_linear(x), x

    def parameter_noise(self):
        for p in self.parameters():
            p.data = torch.normal(
                means=p.data,
                std=p.data.abs()*parameter_noise_rate,
                )

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
