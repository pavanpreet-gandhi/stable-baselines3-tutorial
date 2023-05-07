import gym
from stable_baselines3 import A2C
import os

env = gym.make('LunarLander-v2')

# Create necessary directories
models_dir = 'models/A2C'
log_dir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Train model - save and log along the way
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10_000 # number of timesteps between saves
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='A2C')
    model.save(f'{models_dir}/{TIMESTEPS*i}')