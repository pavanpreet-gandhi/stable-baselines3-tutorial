import gym
from stable_baselines3 import A2C
import os, time

env = gym.make('LunarLander-v2')

# Create necessary directories
start_time_id = int(time.time()) # seconds since January 1, 1970, 00:00:00 (UTC)
models_dir = f'models/A2C-{start_time_id}'
log_dir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Train model - save and log along the way
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10_000 # number of timesteps between saves
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'A2C-{start_time_id}')
    model.save(f'{models_dir}/{TIMESTEPS*i}')