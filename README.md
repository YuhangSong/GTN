# GTN based on A2C

This is a PyTorch implementation of GTN based on A2C.

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.
 
## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
<<<<<<< HEAD
# virtual env
source ~/.bashrc && source deactivate && conda remove --name gtn_env --all
conda create -n gtn_env && source ~/.bashrc && source activate gtn_env
=======
# clear env
source ~/.bashrc && source deactivate && conda remove --name gtn_env --all

# create
conda create -n gtn_env

# source in
source ~/.bashrc
source activate gtn_env

# clear dir
rm -r gtn_env

# create dir
mkdir -p gtn_env/project/ && cd gtn_env/project/
>>>>>>> 4d6b796391ba55478b23f586b832e95ee6029c87

# PyTorch
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 
pip install torchvision

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

# Other requirements
git clone https://github.com/YuhangSong/gtn_a2c.git
cd gtn_a2c
pip install -r requirements.txt
cd ..
```

## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request. Also see a todo list below.

### TODO
* Improve this README file. Rearrange images.

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.

### Atari
#### GTN based on A2C

```bash
<<<<<<< HEAD
source ~/.bashrc && source activate gtn_env && CUDA_VISIBLE_DEVICES=1 python main.py --env-name "PongNoFrameskip-v4"
=======
source ~/.bashrc && source activate gtn_env
CUDA_VISIBLE_DEVICES=0 python main.py
>>>>>>> 4d6b796391ba55478b23f586b832e95ee6029c87
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --num-processes 8 --num-steps 256 --vis-interval 1 --log-interval 1
```

#### ACKTR

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

### MuJoCo
#### A2C

```bash
python main.py --env-name "Reacher-v1" --num-stack 1 --num-frames 1000000
```

#### PPO

```bash
python main.py --env-name "Reacher-v1" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --batch-size 64 --gamma 0.99 --tau 0.95 --num-frames 1000000
```

#### ACKTR

ACKTR requires some modifications to be made specifically for MuJoCo. But at the moment, I want to keep this code as unified as possible. Thus, I'm going for better ways to integrate it into the codebase.

## Results

### XX

![BreakoutNoFrameskip-v4](imgs/a2c_breakout.png)
