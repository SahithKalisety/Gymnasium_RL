import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render = False):

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode = 'human')

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array

    lr = 0.9 # learning rate
    df = 0.9 # discount factor Near 0: more weight/reward placed on immediate state. Near 1: more on future state.

    #epsilon greedy policy

    epsilon = 1  #1 = 100% random actions
    epsilon_df = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):

        state = env.reset()[0] # states: 0 to 63, 0=top left corner, 63=bottom right corner
        terminated = False #Becomes true when fall in hole or eached goal
        truncated= False #True when the actions reach mare than 150 steps

        while(not truncated  and not terminated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            action = env.action_space.sample() #actions: 0 = left, 1 = down, 2 = right, 3 = up

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + lr * (reward + df + np.max(q[new_state,:])-q[state,action])

            state = new_state

        epsilon = max(epsilon - epsilon_df, 0)

        if(epsilon==0):
            lr = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':

    run(1000, is_training=True, render=True)