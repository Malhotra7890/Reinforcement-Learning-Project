This project implements a Deep Deterministic Policy Gradient (DDPG) agent to solve LunarLanderContinuous-v3 from Gymnasium.
It includes complete modules for training, evaluation, visualization, and analysis of the agent’s performance.

Important Note for Readers

Some training logs and full JSON files exceeded GitHub’s 25MB upload limit, so only sample data is included for demonstration.
If you need the full dataset or complete logs, you can:

Regenerate them by running train_ddpg.py, or


Project Structure (and Purpose of Each File)
 Core Modules
File	Purpose (Brief)
ddpg_agent.py	Implements the full DDPG algorithm—Actor/Critic neural networks, Replay Buffer, action selection, gradient updates, and soft target updates. This is the intelligence core of the agent.
train_ddpg.py	Controls the entire training loop. Interacts with the environment, collects transitions, updates the model, logs rewards, and saves trained weights.
visualize_ddpg.py	Loads saved models and runs a visual simulation using render_mode="human". Used to demonstrate the final lunar landing behavior.
LunarLander_DDPG_Analysis.ipynb	Analyzes training logs, plots reward curves, velocity patterns, thrust usage, and other behavior metrics. Useful for understanding learning progression.
Saved Model Files
File	Description
ddpg_actor.pth	Final trained Actor network used for action prediction.
ddpg_critic.pth	Final trained Critic network used for Q-value estimation.

Log / Sample Data
File	Description
ddpg_training_rewards.txt	Stores episode-wise total rewards during training.
ddpg_training_data_sample.json	A reduced sample dataset containing state, action, and reward parameters (full 25MB data omitted).


Algorithms & Techniques Used

Actor–Critic architecture

Soft target network updates

Replay buffer for stable learning

Exploration noise for continuous actions

PyTorch neural network modeling
