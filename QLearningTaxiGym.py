import gym, random, time, sys
import numpy as np
env = gym.make("Taxi-v3").env
env.reset()

q_table = np.zeros([env.observation_space.n, env.action_space.n])
# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
total_epochs, total_penalties = 0, 0
episodes = 5000

for i in range(episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)
        # print(f"Action: {action}")
        # print(f"Current State: {state}")
        # print(f"Next state: {next_state}")
        old_value = q_table[state, action]
        # print(f"Old value: {old_value}")
        next_max = np.max(q_table[next_state])
        # print(f"Next max: {next_max}")
        # print(f"Reward: {reward}")

        # print("#"*40)

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1


    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")





