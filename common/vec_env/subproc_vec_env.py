import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv

def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class Mt_SubprocVecEnv(object):
    """docstring for Mt_SubprocVecEnv"""
    def __init__(self, envs):
        super(Mt_SubprocVecEnv, self).__init__()
        self.envs = envs

        action_space_n = 0
        for env in self.envs:
            if env.action_space.n > action_space_n:
                self.action_space = env.action_space
                action_space_n = env.action_space.n

        self.observation_space = self.envs[0].observation_space

    def step(self, actions):
        return_list_0 = []
        return_list_1 = []
        return_list_2 = []
        for ii in range(len(self.envs)): 
            temp_0, temp_1, temp_2, _ = self.envs[ii].step(
                np.clip(
                    actions[self.envs[ii].num_process*ii:self.envs[ii].num_process*(ii+1)],
                    a_min=0,
                    a_max=self.envs[ii].action_space.n-1))
            return_list_0 += [temp_0]
            return_list_1 += [temp_1]
            return_list_2 += [temp_2]

        return_concatenated_0 = np.concatenate(return_list_0, axis=0)
        return_concatenated_1 = np.concatenate(return_list_1, axis=0)
        return_concatenated_2 = np.concatenate(return_list_2, axis=0)

        return return_concatenated_0, return_concatenated_1, return_concatenated_2

    def reset(self):
        return_list = []
        for env in self.envs:
            return_list += [env.reset()]
        return_concatenated = np.concatenate(return_list, axis=0)
        return return_concatenated

    def close(self):
        for env in self.envs:
            env.close()

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.num_process = nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])        
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn))) 
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)
