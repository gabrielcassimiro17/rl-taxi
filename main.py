from agent import Agent
from monitor import interact
import gym
from collections import defaultdict
import pickle
import numpy as np

env = gym.make('Taxi-v3')  # render_mode='human'
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent, 50_000)

agent.Q = dict(agent.Q)

with open("agent.pkl", "wb") as f:
    pickle.dump(agent, f, fix_imports=True)
