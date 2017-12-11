import copy
import glob
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from common.vec_env.subproc_vec_env import SubprocVecEnvMt
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

from arguments import debugging, gtn_M
from arguments import exp, title, title_html
is_use_afs = True
args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.algo == 'ppo':
    assert args.num_processes * args.num_steps % args.batch_size == 0
'''num_frames: number of frames to train (default: 10e6)
    num_steps:  agent every time updata need steps
'''
num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

mt_env_id_dic_all = {
    'mt test pong':[
        'PongNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        ],
    'mt high performance':[
        'BeamRiderNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'PongNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        ],
    'mt as ewc test':[
        'CrazyClimberNoFrameskip-v4',
        'RiverraidNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'PongNoFrameskip-v4',
        'StarGunnerNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'AsteroidsNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        ],
    'mt shooting':[
        'BeamRiderNoFrameskip-v4',
        'PhoenixNoFrameskip-v4',
        'AtlantisNoFrameskip-v4',
        'CentipedeNoFrameskip-v4',
        'RiverraidNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'GravitarNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsteroidsNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        'YarsRevengeNoFrameskip-v4',
        'CarnivalNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
        'ZaxxonNoFrameskip-v4',
        'PooyanNoFrameskip-v4',
        'StarGunnerNoFrameskip-v4',
        ],
    'mt all atari':[
        'CarnivalNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'AmidarNoFrameskip-v4',
        'BankHeistNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'TutankhamNoFrameskip-v4',
        'VentureNoFrameskip-v4',
        'WizardOfWorNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsteroidsNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'CentipedeNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
        'DemonAttackNoFrameskip-v4',
        'AtlantisNoFrameskip-v4',
        'GravitarNoFrameskip-v4',
        'PhoenixNoFrameskip-v4',
        'PooyanNoFrameskip-v4',
        'RiverraidNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        'StarGunnerNoFrameskip-v4',
        'TimePilotNoFrameskip-v4',
        'ZaxxonNoFrameskip-v4',
        'YarsRevengeNoFrameskip-v4',
        'AsterixNoFrameskip-v4',
        'ElevatorActionNoFrameskip-v4',
        'BerzerkNoFrameskip-v4',
        'FreewayNoFrameskip-v4',
        'FrostbiteNoFrameskip-v4',
        'JourneyEscapeNoFrameskip-v4',
        'KangarooNoFrameskip-v4',
        'KrullNoFrameskip-v4',
        'PitfallNoFrameskip-v4',
        'SkiingNoFrameskip-v4',
        'UpNDownNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'RoadRunnerNoFrameskip-v4',
        'DoubleDunkNoFrameskip-v4',
        'IceHockeyNoFrameskip-v4',
        'MontezumaRevengeNoFrameskip-v4',
        'GopherNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'PongNoFrameskip-v4',
        'PrivateEyeNoFrameskip-v4',
        'TennisNoFrameskip-v4',
        'VideoPinballNoFrameskip-v4',
        'FishingDerbyNoFrameskip-v4',
        'NameThisGameNoFrameskip-v4',
        'BowlingNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4',
        'BoxingNoFrameskip-v4',
        'JamesbondNoFrameskip-v4',
        'RobotankNoFrameskip-v4',
        'SolarisNoFrameskip-v4',
        'EnduroNoFrameskip-v4',
        'KungFuMasterNoFrameskip-v4',
        ],
}

mt_env_id_dic_selected = mt_env_id_dic_all[args.env_name]

for env_id in mt_env_id_dic_selected:
    log_dir = args.log_dir+env_id+'/'
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.json'))
        for f in files:
            os.remove(f)

afs_offset = [0.0, 0.0, 0.0, 0.0, 0.0]

def main():
    print("#######")
    print("WARNING: All rewards are clipped so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = []
        for i in range(len(mt_env_id_dic_selected)):
            win += [None]
        win_afs_per_m = None
        win_afs_loss = None
        win_basic_loss = None

    envs = []
    ''' Because the oral program has only one game per model, so Song add loop i
        So whatever you wanna run , just put in SubprocVecEnvMt!
    '''
    for i in range(len(mt_env_id_dic_selected)):
        log_dir = args.log_dir+mt_env_id_dic_selected[i]+'/'
        for j in range(args.num_processes):
            envs += [make_env(mt_env_id_dic_selected[i], args.seed, j, log_dir)]
    ''' This envs is an intergration of all the running env'''
    envs = SubprocVecEnvMt(envs)

    num_processes_total = args.num_processes * len(mt_env_id_dic_selected)
    '''(1,128,128)'''
    obs_shape = envs.observation_space.shape
    #num_stack :number of frames to stack
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    from arguments import is_restore
    if is_restore and args.save_dir:
        load_path = os.path.join(args.save_dir, args.algo)
        actor_critic =torch.load(os.path.join(load_path, args.env_name + ".pt"))
        print ("restored previous model!")
        print (actor_critic.Variable)
        print (sss)
    else:
        if len(envs.observation_space.shape) == 3:
            actor_critic = CNNPolicy(obs_shape[0], envs.action_space)
        else:
            actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
    
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)
    #'args.num_steps: number of forward steps in A2C
    #rollouts is an intergration of state\ reward\ next state\action and so on
    rollouts = RolloutStorage(args.num_steps, num_processes_total, obs_shape, envs.action_space)
    current_state = torch.zeros(num_processes_total, *obs_shape)
    ''' not sure about it'''
    def update_current_state(state):
        shape_dim0 = envs.observation_space.shape[0]
        # print (shape_dim0)
        # print (sss)
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    state = envs.reset()
    update_current_state(state)

    rollouts.states[0].copy_(current_state)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes_total, 1])
    final_rewards = torch.zeros([num_processes_total, 1])

    if args.cuda:
        current_state = current_state.cuda()
        rollouts.cuda()

    if args.algo == 'ppo':
        old_model = copy.deepcopy(actor_critic)

    from arguments import ewc, ewc_lambda, ewc_interval

    afs_per_m = []
    afs_offset = [0.0]*gtn_M

    afs_loss_list = []
    basic_loss_list = []

    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    '''for one whole game '''
    for j in range(num_updates):
        for step in range(args.num_steps):
            if ewc == 1:
                try:
                    states_store = torch.cat([states_store, rollouts.states[step].clone()], 0)
                except Exception as e:
                    states_store = rollouts.states[step].clone()
            # Sample actions
            '''act fun refer to "observe it!"'''
            value, action = actor_critic.act(Variable(rollouts.states[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next state
            state, reward, done = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_state.dim() == 4:
                current_state *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_state *= masks

            update_current_state(state)
            rollouts.insert(step, current_state, action.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *obs_shape))

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            # reset gradient
            optimizer.zero_grad()

            # forward
            values, action_log_probs, dist_entropy, conv_list = actor_critic.evaluate_actions(Variable(rollouts.states[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))
            # pre-process
            values = values.view(args.num_steps, num_processes_total, 1)
            action_log_probs = action_log_probs.view(args.num_steps, num_processes_total, 1)

            # compute afs loss
            afs_per_m_temp, afs_loss = actor_critic.get_afs_per_m(
                action_log_probs=action_log_probs,
                conv_list=conv_list,
            )
            if len(afs_per_m_temp)>0:
                afs_per_m += [afs_per_m_temp]

            if (afs_loss is not None) and (afs_loss.data.cpu().numpy()[0]!=0.0):
                afs_loss.backward(mone, retain_graph=True)
                afs_loss_list += [afs_loss.data.cpu().numpy()[0]]

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()
            final_loss_basic = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef

            ewc_loss = None
            if j != 0:
                if ewc == 1:
                    ewc_loss = actor_critic.get_ewc_loss(lam=ewc_lambda)
            
            if ewc_loss is None:
                final_loss = final_loss_basic
            else:
                final_loss = final_loss_basic + ewc_loss
            # print (final_loss_basic.data.cpu().numpy()[0])
            # final_loss_basic
            basic_loss_list += [final_loss_basic.data.cpu().numpy()[0]]
            final_loss.backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            old_model.load_state_dict(actor_critic.state_dict())
            if hasattr(actor_critic, 'obs_filter'):
                old_model.obs_filter = actor_critic.obs_filter

            for _ in range(args.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(num_processes_total * args.num_steps)), args.batch_size * num_processes_total, drop_last=False)
                for indices in sampler:
                    indices = torch.LongTensor(indices)
                    if args.cuda:
                        indices = indices.cuda()
                    states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
                    actions_batch = rollouts.actions.view(-1, action_shape)[indices]
                    return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, conv_list = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

                    _, old_action_log_probs, _, old_conv_list= old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
                    adv_targ = Variable(advantages.view(-1, 1)[indices])
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    final_loss_basic = (value_loss + action_loss - dist_entropy * args.entropy_coef)
                    
                    basic_loss_list += [final_loss_basic.data.cpu().numpy()[0]]
                    final_loss_basic.backward()
                    optimizer.step()

        rollouts.states[0].copy_(rollouts.states[-1])

        if j % int(num_updates/2-10) == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, (j + 1) * args.num_processes * args.num_steps,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))

            try:
                print("ewc loss {:.5f}".
                format(ewc_loss.data.cpu().numpy()[0]))
            except Exception as e:
                pass
            

        if j > 5 and j % args.vis_interval == 0 and args.vis:
            ''' load from the folder'''
            for ii in range(len(mt_env_id_dic_selected)):
                log_dir = args.log_dir+mt_env_id_dic_selected[ii]+'/'
                win[ii] = visdom_plot(viz, win[ii], log_dir, mt_env_id_dic_selected[ii], args.algo)

            if len(afs_per_m)>0:
                win_afs_per_m = viz.line(
                    torch.from_numpy(np.asarray(afs_per_m)), 
                    win=win_afs_per_m,
                    opts=dict(title=title_html+'>>afs')
                )

            # print (basic_loss_list)
            '''a2c:len(basic_loss_list) is vis_interval+1. because j start from 0
               ppo:len(basic_loss_list) is (vis_interval+1)*ppo_epoch_4*len(BatchSampler)
            '''
            
            # print (len(basic_loss_list))
            # print (ss)
            win_basic_loss = viz.line(
                torch.from_numpy(np.asarray(basic_loss_list)), 
                win=win_basic_loss,
                opts=dict(title=title_html+'>>basic_loss')
            )

            if len(afs_loss_list) > 0:
                win_afs_loss = viz.line(
                    torch.from_numpy(np.asarray(afs_loss_list)), 
                    win=win_afs_loss,
                    opts=dict(title=title_html+'>>afs_loss')
                )

        from arguments import parameter_noise, parameter_noise_interval
        if parameter_noise == 1:
            if j % parameter_noise_interval == 0:
                actor_critic.parameter_noise()

        if ewc == 1:
            if j % ewc_interval == 0 or j==0:
                actor_critic.compute_fisher(states_store)
                states_store = None
                actor_critic.star()

if __name__ == "__main__":
    main()
