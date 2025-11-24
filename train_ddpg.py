import gymnasium as gym
import torch
from ddpg_agent import DDPGAgent
import datetime
import os
import json
import numpy as np

EPISODES = 1500
MAX_STEPS = 850
REWARD_LOG_FILE = "ddpg_training_rewards.txt"
TRAINING_DATA_FILE = "ddpg_training_data.json"

print("========Training started========")
env = gym.make("LunarLanderContinuous-v3")
agent = DDPGAgent()  # make sure agent has proper hidden layers (256x2 recommended)

# Prepare text file heading with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
header = f"\n===== Training Session at {timestamp} =====\n"
with open(REWARD_LOG_FILE, "a") as f:
    f.write(header)

# Container for step-wise data
all_episodes_data = []

for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    # Step-wise lists
    main_thrusts = []
    side_thrusts = []
    velocity_x = []
    velocity_y = []
    angle = []
    angular_velocity = []
    position_x = []
    position_y = []

    for _ in range(MAX_STEPS):
        # Add exploration noise during training
        action = agent.select_action(state)  # agent internally adds noise
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Record step-wise parameters
        main_thrusts.append(float(action[0]))
        side_thrusts.append(float(action[1]))
        velocity_x.append(float(state[2]))
        velocity_y.append(float(state[3]))
        angle.append(float(state[4]))
        angular_velocity.append(float(state[5]))
        position_x.append(float(state[0]))
        position_y.append(float(state[1]))

        agent.train()
        if done:
            break

    # Log episode reward
    print(f"Episode {ep} | Reward = {total_reward:.2f}")
    with open(REWARD_LOG_FILE, "a") as f:
        f.write(f"Episode {ep}: Reward = {total_reward:.2f}\n")

    # Store step-wise data for plotting
    episode_data = {
        "reward": float(total_reward),
        "main_thrusts": main_thrusts,
        "side_thrusts": side_thrusts,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "angle": angle,
        "angular_velocity": angular_velocity,
        "position_x": position_x,
        "position_y": position_y
    }
    all_episodes_data.append(episode_data)

# Save step-wise data to JSON
with open(TRAINING_DATA_FILE, "w") as f:
    json.dump(all_episodes_data, f, indent=2)

# Save model
torch.save(agent.actor.state_dict(), "ddpg_actor.pth")
torch.save(agent.critic.state_dict(), "ddpg_critic.pth")

print("=======Training completed=======")
print("Model saved")
print(f"Step-wise training data saved to {TRAINING_DATA_FILE}")

