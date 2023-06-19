import gymnasium as gym
import random

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
discount = 1
theta = 0.001

random.seed(0)
env.reset(seed=0)

no_of_actions = env.env.action_space.n
actions = range(0, env.env.action_space.n)
states = range(0, env.env.observation_space.n)
tp_matrix = env.env.P

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
print(env.render())

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
            action = max(policy[state], key=policy[state].get)
        actions.append(action)
        state, reward, done, _, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
    return states, actions, rewards

def policy_evaluation(V, policy, discount, theta, tp_matrix):
    while True:
        delta = 0
        for s in states:
            v = 0
            for a in actions:
                for p, s_, r, _ in tp_matrix[s][a]:
                    v += policy[s][a] * p * (r + discount * V[s_])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
            
        if delta < theta:
            break
    return V

def policy_improvement(V, policy, discount, tp_matrix):
    V = policy_evaluation(V, policy, discount, theta, tp_matrix)
    
    policy_stable = True
    for s in states:
        old_policy = policy[s]
        action_values = [0 for a in actions]
        for a in actions:
            for p, s_, r, _ in tp_matrix[s][a]:
                action_values[a] += p * (r + discount * V[s_])
                
        max_action_value = max(action_values)
        amount_of_max_actions = action_values.count(max_action_value)
        
        policy[s] = {a: 1/amount_of_max_actions if a in [i for i, v in enumerate(action_values) if v == max_action_value] else 0 for a in actions}
        
        if old_policy != policy[s]:
            policy_stable = False
        
    if policy_stable:
        return V, policy
    else: 
        return policy_improvement(V, policy, discount, tp_matrix)
        

# put your solution here
def main():
    value_function = {s: 0 for s in states}
    policy = {s: {a: 0 for a in actions} for s in states}
    
    optimal_V, optimal_policy = policy_improvement(value_function, policy, discount, tp_matrix)
    print(optimal_V)
    print(optimal_policy)
    
    print("##############")
    print("###  Try   ###")
    print("##############")
    
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
    runs = 0
    while True:
        runs += 1
        e_states, e_actions, e_rewards = play_episode(env, optimal_policy)
        if sum(e_rewards) > 0:
            break
    print(f"Needed {runs} episodes, random policy took {len(e_states)} steps.")
    print(f"States: {e_states}")
    print(f"Actions: {e_actions}")
    print(f"Rewards: {e_rewards}")
    
    
    

if __name__ == "__main__":
    main()




