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
        self.envs[1].step(actions)
        return self.envs[0].step(actions)

    def reset(self):
        self.envs[1].reset()
        return self.envs[0].reset()

    def close(self):
        for env in self.envs:
            env.close()

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
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
