import gym
from stable_baselines3 import A2C, PPO

env = gym.make('LunarLander-v2')

# Loading model
models_dir = 'models\PPO-1683448504' # best model
model_number = 990_000 # timestep with best reward and shortest episode length
model_path = f'{models_dir}/{model_number}.zip'
model = PPO.load(model_path, env=env)

# Visualizing model performance
episodes = 5
for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs) # pass observation to model to get predicted action
		obs, reward, done, info = env.step(action) # pass action to env and get info back
		env.render() # show the environment on the screen
env.close()