import gymnasium as gym
import torch
from ddpg_agent import DDPGAgent
import pygame
import time

# Initialize environment
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
agent = DDPGAgent()

# Load trained weights
agent.actor.load_state_dict(torch.load("ddpg_actor.pth"))
agent.critic.load_state_dict(torch.load("ddpg_critic.pth"))

NUM_SIMULATIONS = 30
SUCCESS_THRESHOLD = 200

for ep in range(NUM_SIMULATIONS):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, noise=0)  # deterministic
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    # Only display success message on Pygame screen if successful
    if total_reward >= SUCCESS_THRESHOLD:
        # Pygame overlay
        screen = pygame.display.get_surface()
        font = pygame.font.SysFont(None, 60)
        text_surface = font.render("Successful Landing!", True, (0, 255, 0))
        screen.blit(text_surface, (100, 50))
        pygame.display.update()
        time.sleep(4)  # display message 
