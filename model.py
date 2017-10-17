import torch
import torch.nn as nn
import torch.nn.functional as F
from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self, process_per_game):
        super(FFPolicy, self).__init__()
        self.process_per_game = process_per_game

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):

        value, x = self(inputs)
        action = []
        for dist_i in range(len(self.dist)):
            x_temp = x.narrow(0,dist_i*self.process_per_game,self.process_per_game)

            action_temp = self.dist[dist_i].sample(x_temp, deterministic=deterministic)

            action += [action_temp]

        action = torch.cat(action,0)
        
        return value, action

    def evaluate_actions(self, inputs, actions, num_steps):

        value, x = self(inputs) 
        
        # action_log_probs = []
        # dist_entropy = []
        # for dist_i in range(len(self.dist)):
            
        #     action_log_probs_temp, dist_entropy_temp = self.dist[dist_i].evaluate_actions(x.narrow(0,dist_i*self.process_per_game*num_steps,self.process_per_game*num_steps), actions.narrow(0,dist_i*self.process_per_game*num_steps,self.process_per_game*num_steps))
        #     action_log_probs += [action_log_probs_temp]
        #     dist_entropy += [dist_entropy_temp]

        # action_log_probs = torch.cat(action_log_probs,0)
        # dist_entropy = torch.cat(dist_entropy,0)
        # dist_entropy = dist_entropy.mean()

        action_log_probs, dist_entropy = self.dist[0].evaluate_actions(x, actions)

        return value, action_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, game_total, action_space_list, process_per_game):
        super(CNNPolicy, self).__init__(process_per_game)
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, 1)

        # if action_space.__class__.__name__ == "Discrete":
        self.dist = []
        for i in range(game_total):
            self.dist += [Categorical(512, action_space_list[i])]
            # num_outputs = action_space.n
            # self.dist = Categorical(512, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian(512, num_outputs)
        # else:
        #     raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def cuda(self, **args):
        super(CNNPolicy, self).cuda(**args)
        for dist_i in self.dist:
            dist_i=dist_i.cuda()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        return self.critic_linear(x), x


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
