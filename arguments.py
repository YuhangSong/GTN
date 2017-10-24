import argparse
import torch

debugging = 1

gtn_M = 5
gtn_N = 2

hierarchical = 1

parameter_noise = 1
parameter_noise_interval = 10
parameter_noise_rate = 0.01

ewc = 0
ewc_lambda = 1.0
ewc_interval = 10

both_side_tower = 0

log_fisher_sensitivity_per_m = 1
loss_fisher_sensitivity_per_m = 1
if loss_fisher_sensitivity_per_m == 1:
    log_fisher_sensitivity_per_m = 1

# dataset = 'mt shooting'
dataset = 'mt test pong'
# dataset = 'mt all atari'

exp = '1'
exp += ('gtn_1'+'_')
exp += (str(gtn_M)+'x'+str(gtn_N)+'_')
exp += ('hierarchical_'+str(hierarchical)+'_')
exp += ('parameter_noise_'+str(parameter_noise)+'_')
exp += ('ewc_'+str(ewc)+'_')
exp += ('both_side_tower_'+str(both_side_tower)+'_')
exp += ('dataset_'+dataset+'_')

print('#######################################')
print(exp)
print('#######################################')

multi_gpu = 0

if multi_gpu == 1:
    gpus = range(torch.cuda.device_count())
    print('Using GPU:'+str(gpus))
else:
    gpus = [0]

num_processes = 16
if dataset == 'mt all atari':
    num_processes = 4

log_interval = 20
vis_interval = 20

if debugging == 1:
    num_processes = 1
    dataset = 'mt test pong'
    log_interval = 1
    vis_interval = 1

title = exp
for x in range(0,len(title),40):
    title = title[:x] + '\n' + title[x:]
title_html = exp
for x in range(0,len(title),40):
    title_html = title_html[:x] + '<br>' + title_html[x:]

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=num_processes,
                        help='how many training CPU processes to use')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ppo batch size (default: 64)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=log_interval,
                        help='log interval, one log per n updates')
    parser.add_argument('--vis-interval', type=int, default=vis_interval,
                        help='vis interval, one log per n updates')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default=dataset,
                        help='environment to train on')
    parser.add_argument('--log-dir', default='../../result/'+exp+'/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='../../result/'+exp+'/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
