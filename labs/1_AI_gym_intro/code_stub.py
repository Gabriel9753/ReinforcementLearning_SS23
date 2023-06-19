import gymnasium as gym
import random
from tqdm import tqdm
from collections import defaultdict

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

random.seed(0)
env.reset(seed=0)

print("## Frozen Lake ##")
print("Start state:")
print(env.render())

no_of_actions = env.env.action_space.n
action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

def play_episode(env, policy=None):
    done = False
    state, _ = env.reset()
    states = [state]
    actions = []
    rewards = []
    while not done:
        if policy is None:
            action = random.randint(0, no_of_actions-1)
        else:
            action = policy[state]
        actions.append(action)
        state, reward, done, _, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
    return states, actions, rewards

def main():
    global no_of_actions, action2string, env
    ## TASK 1 ##
    print("##############")
    print("### TASK 1 ###")
    print("##############")
    runs = 0
    while True:
        runs += 1
        states, actions, rewards = play_episode(env)
        if sum(rewards) > 0:
            break
    print(f"Needed {runs} episodes, random policy took {len(states)} steps.")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    
    policy = defaultdict(int)
    for i, v in enumerate(states[:-1]):
        policy[v] = actions[i]
    print(f"New policy: {policy}") 
    
    states, actions, rewards = play_episode(env, policy)
    if sum(rewards) > 0:
        print(f"Success: new policy needed {len(states)} steps")
    else:
        print(f"New policy failed!")
        
    ## TASK 2 ##
    print("##############")
    print("### TASK 2 ###")
    print("##############")
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    runs = 0
    while True:
        runs += 1
        states, actions, rewards = play_episode(env)
        if sum(rewards) > 0:
            break
    print(f"Needed {runs} episodes, random policy took {len(states)} steps.")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    
    policy = defaultdict(int)
    for i, v in enumerate(states[:-1]):
        policy[v] = actions[i]
    print(f"New policy: {policy}") 
    
    states, actions, rewards = play_episode(env, policy)
    if sum(rewards) > 0:
        print(f"Success: new policy needed {len(states)} steps")
    else:
        print(f"New policy failed!")
        
    ## TASK 3 ##
    print("##############")
    print("### TASK 3 ###")
    print("##############")
    
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
    runs = 0
    while True:
        runs += 1
        states, actions, rewards = play_episode(env)
        if sum(rewards) > 0:
            break
    print(f"Needed {runs} episodes, random policy took {len(states)} steps.")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    
    # -> policy is not working anymore because of the slippery floor -> need to learn a new policy
    
if __name__ == "__main__":
    main()


# state, _ = env.reset()
# done = False

# while not done:
#     action = random.randint(0, no_of_actions-1)  # choose a random action
#     state, reward, done, _, _ = env.step(action)
#     print(f"\nAction:{action2string[action]}, new state:{state}, reward:{reward}")
#     print(env.render())
