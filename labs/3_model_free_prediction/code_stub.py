import random
import gymnasium as gym
import numpy as np
from prettytable import PrettyTable
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", render_mode="ansi")
random.seed(0)
np.random.seed(0)
env.reset(seed=0)

print("## Frozen Lake ##")
print("Start state:")
print(env.render())

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

no_states = env.observation_space.n
no_actions = env.action_space.n

def play_episode(q_values=None, epsilon=None):
    state, _ = env.reset()
    done = False
    r_s = []
    s_a = []
    while not done:
        if q_values is None:
            action = random.randint(0, 3)
        elif epsilon is not None:
            action = epsilon_greedy(q_values, epsilon)[state]
        else:
            argmax_values = np.argmax(q_values, axis=1)
            action = argmax_values[state]
        s_a.append((state, action))
        state, reward, done, _, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s

def update_q_values(q_values, q_counter, rewards, s_a):
    # Task 1: update Q-values using MC
    for i, (s,a) in enumerate(s_a):
        return_i = sum(rewards[i:])
        q_counter[s][a] += 1
        q_values[s][a] += (1/q_counter[s][a]) * (return_i - q_values[s][a])
    return q_values

def epsilon_greedy(q_values, epsilon=0.1):
    argmax_values = np.argmax(q_values, axis=1)
    greedy_actions = np.random.binomial(1, 1-epsilon, size=argmax_values.shape[0])
    random_actions = np.random.randint(0, no_actions, size=argmax_values.shape[0])
    return np.where(greedy_actions, argmax_values, random_actions)

def print_q_values(q_values):
    table = PrettyTable()
    table.field_names = ["State", "Left", "Down", "Right", "Up"]
    for s in range(no_states):
        values = [round(q_values[s][a],2) for a in range(no_actions)]
        max_val = max(values)
        color_values = [colored(val, 'red', None, ['bold', 'blink']) if val == max_val else val for val in values]
        table.add_row([colored(s, 'green'), *color_values])
    print(table)

def test_policy(q_values, episodes=100):
    all_rewards = 0
    for i in range(episodes):
        _, r = play_episode(q_values)
        all_rewards += sum(r)
    return all_rewards/episodes

def train_mc(succesful_episodes=100, epsilon=None, random=False):
    losses = []
    total_runs = 0
    q_values = np.zeros((no_states, no_actions))
    q_counter = np.zeros((no_states, no_actions))
    with tqdm(desc=f"Training with random: {random} and epsilon: {epsilon}", total=succesful_episodes) as pbar:
        while succesful_episodes > 0:
            s_a, rewards = play_episode(None if epsilon is None else q_values, epsilon=epsilon)
            q_values = update_q_values(q_values, q_counter, rewards, s_a)
            total_runs += 1
            if sum(rewards) > 0:
                sum_rewards = test_policy(None if random else q_values)
                pbar.set_postfix_str(f"Total runs: {total_runs} | Average reward: {sum_rewards:.2f}")
                losses.append(sum_rewards)
                succesful_episodes -= 1
                pbar.update(1)
    return losses

def main():
    successful_episodes = 2000
    # total_runs = 0
    random_loss = train_mc(successful_episodes, epsilon=None, random=True)
    greedy_loss = train_mc(successful_episodes, epsilon=None, random=False)
    epsilon_greedy_loss = train_mc(successful_episodes, epsilon=0.1, random=False)
    epsilon_greedy_loss_2 = train_mc(successful_episodes, epsilon=0.2, random=False)
    epsilon_greedy_loss_3 = train_mc(successful_episodes, epsilon=0.5, random=False)
    epsilon_greedy_loss_4 = train_mc(successful_episodes, epsilon=0.7, random=False)
    epsilon_greedy_loss_5 = train_mc(successful_episodes, epsilon=0.9, random=False)
                
    def moving_average(data, window_size):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    window_size = 20  # choose a window size that's appropriate for your data
    # Apply moving average to data
    losses_smooth = moving_average(greedy_loss, window_size)
    losses_random_smooth = moving_average(random_loss, window_size)
    epsilon_greedy_loss_smooth = moving_average(epsilon_greedy_loss, window_size)
    epsilon_greedy_loss_2_smooth = moving_average(epsilon_greedy_loss_2, window_size)
    epsilon_greedy_loss_3_smooth = moving_average(epsilon_greedy_loss_3, window_size)
    epsilon_greedy_loss_4_smooth = moving_average(epsilon_greedy_loss_4, window_size)
    epsilon_greedy_loss_5_smooth = moving_average(epsilon_greedy_loss_5, window_size)

    # print_q_values(q_values)
    plt.plot(losses_smooth, label="Greedy")
    plt.plot(losses_random_smooth, label="Random")
    plt.plot(epsilon_greedy_loss_smooth, label="Epsilon (0.1) Greedy")
    plt.plot(epsilon_greedy_loss_2_smooth, label="Epsilon (0.2) Greedy")
    plt.plot(epsilon_greedy_loss_3_smooth, label="Epsilon (0.5) Greedy")
    plt.plot(epsilon_greedy_loss_4_smooth, label="Epsilon (0.7) Greedy")
    plt.plot(epsilon_greedy_loss_5_smooth, label="Epsilon (0.9) Greedy")
    
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Episodes")
    plt.legend()
    plt.show()


main()
