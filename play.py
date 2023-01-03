import pickle
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

trained_agent_video = "trained_agent_video.mp4"


with open("agent.pkl", "rb") as f:
    agent = pickle.load(f)

env = gym.make('Taxi-v3', render_mode='rgb_array')
video = VideoRecorder(env, trained_agent_video)

state, _ = env.reset()


sum_of_rewards = 0

while True:
    env.render()
    video.capture_frame()
    action = agent.select_best_action(state)
    state, reward, done, _, _ = env.step(action)
    sum_of_rewards += reward

    if done:
        break

video.close()
env.close()