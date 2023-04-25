import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt
from pettingzoo.atari import space_invaders_v2
from pettingzoo.sisl import pursuit_v4
import gym
from gym.wrappers import RecordVideo
from PIL import Image

class DQN_com(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out = nn.Linear(64,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.out(x)
        return x


class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def eps_decay(itr,num_episodes, eps_start, eps_end):
    if itr>=num_episodes*3000:
        return eps_end
    else:
        return (eps_end - eps_start)*itr/(num_episodes*3000) + eps_start

def action(observation, eps, n_actions, policy_net):
    p = np.random.random()
    if p < eps:
        a = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
    else:
        a = policy_net(observation).max(1)[1].view(1, 1)

    return a


def update_network_com(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma):
    if len(memory) < batch_size:
        return

    batch_data = memory.sample(batch_size)

    batch = Transition(*zip(*batch_data))

    #state_batch = torch.cat([image.unsqueeze(0) for image in batch.state], dim=0)
    state_batch = torch.cat(batch.state)
    #print("State Batch Size:", state_batch.size())
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    dones_batch = torch.tensor([float(b) for b in batch.done])
    #dones_batch = torch.where(batch.done, torch.tensor(1), torch.tensor(0))
    #next_state_batch = torch.cat([image.unsqueeze(0) for image in batch.next_state], dim=0)
    next_state_batch = torch.cat(batch.next_state)
    #print("Next State Batch Size:", next_state_batch.size())

    Q_s_a = policy_net(state_batch).gather(1, action_batch)

    y_js = torch.zeros(batch_size)


    with torch.no_grad():
        y_js = (target_net(next_state_batch).max(1)).values
        y_js = reward_batch + (1-dones_batch)*gamma*y_js

    y_js = y_js.float()


    criterion = nn.MSELoss()

    loss = criterion(Q_s_a, y_js.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_network_com(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions,
                    batch_size, eps_start, eps_end, gamma, Transition):
    reward_array = np.zeros(num_episodes)
    itr = 0
    for i_episode in range(num_episodes):
        print(i_episode,"th is running")
        # Initialize the environment and get it's state
        env.reset()
        time = 0
        termination = False
        truncation = False
        for agent in env.agent_iter():
#             if time%500 == 0:
#                 print("Time:",time)
            eps = eps_decay(itr,num_episodes, eps_start, eps_end)
            observation, r, termination, truncation, info = env.last()
            observation = torch.tensor(observation).to(torch.float)
            observation = (torch.flatten(observation)).unsqueeze(0)
            #observation = observation.permute(2, 0, 1)
            if termination or truncation:
                a = None
            else:
                a = action(observation, eps, n_actions, policy_net)

            env.step(a.item())

            observation_next, r, termination, truncation, info = env.last()

            r = torch.tensor([r])
            reward_array[i_episode] += r
            time += 1

            observation_next = torch.tensor(observation_next).to(torch.float)
            observation_next = (torch.flatten(observation_next)).unsqueeze(0)
            #observation_next = observation_next.permute(2, 0, 1)

            memory.push(observation, a, observation_next, r, termination or truncation)

#             # Move to the next state
#             s = s_next

            # Perform one step of the optimization (on the policy network)
            update_network_com(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma)

            itr+=1

            if itr%5000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if time>=3000:
                break

    return policy_net, reward_array

class DQN_sep(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64, 64)
        self.out = nn.Linear(64,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.out(x)
        return x

def update_network_sep(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma):
    if len(memory) < batch_size:
        return

    for i in range(6):
        batch_data = memory.sample(batch_size)

        batch = Transition(*zip(*batch_data))

        #state_batch = torch.cat([image.unsqueeze(0) for image in batch.state], dim=0)
        state_batch = torch.cat(batch.state)
        #print("State Batch Size:", state_batch.size())
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dones_batch = torch.tensor([float(b) for b in batch.done])
        #dones_batch = torch.where(batch.done, torch.tensor(1), torch.tensor(0))
        #next_state_batch = torch.cat([image.unsqueeze(0) for image in batch.next_state], dim=0)
        next_state_batch = torch.cat(batch.next_state)
        #print("Next State Batch Size:", next_state_batch.size())

        Q_s_a = policy_net[i](state_batch).gather(1, action_batch)

        y_js = torch.zeros(batch_size)


        with torch.no_grad():
            y_js = (target_net[i](next_state_batch).max(1)).values
            y_js = reward_batch + (1-dones_batch)*gamma*y_js

        y_js = y_js.float()


        criterion = nn.MSELoss()

        loss = criterion(Q_s_a, y_js.unsqueeze(1))

        optimizer[i].zero_grad()
        loss.backward()
        optimizer[i].step()

def train_network_sep(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions,
                    batch_size, eps_start, eps_end, gamma, Transition):
    reward_array = np.zeros(num_episodes)
    itr = 0
    for i_episode in range(num_episodes):
        print(i_episode,"th is running")
        # Initialize the environment and get it's state
        env.reset()
        time = 0
        termination = False
        truncation = False
        for agent in env.agent_iter():
#             if time%500 == 0:
#                 print("Time:",time)
            eps = eps_decay(itr,num_episodes, eps_start, eps_end)
            observation, r, termination, truncation, info = env.last()
            observation = torch.tensor(observation).to(torch.float)
            observation = (torch.flatten(observation)).unsqueeze(0)
            #observation = observation.permute(2, 0, 1)
            if termination or truncation:
                a = None
            else:
                a = action(observation, eps, n_actions, policy_net[int(str(agent)[-1])])

            env.step(a.item())

            observation_next, r, termination, truncation, info = env.last()

            r = torch.tensor([r])
            reward_array[i_episode] += r
            time += 1

            observation_next = torch.tensor(observation_next).to(torch.float)
            observation_next = (torch.flatten(observation_next)).unsqueeze(0)
            #observation_next = observation_next.permute(2, 0, 1)

            memory.push(observation, a, observation_next, r, termination or truncation)

#             # Move to the next state
#             s = s_next

            # Perform one step of the optimization (on the policy network)
            update_network_sep(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma)

            itr+=1

            if itr%5000 == 0:
                target_net[i].load_state_dict(policy_net[i].state_dict())

            if time>=3000:
                break

    return policy_net, reward_array

def main():
    batch_size = 32
    gamma = 1
    eps_start = 1
    eps_end = 0.1
    num_episodes = 500
    n_actions = 5

    env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=25,
    n_pursuers=6,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
    catch_reward=5.0, urgency_reward=0, surround=True, constraint_window=1.0)

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    policy_net = DQN_com(3*7*7, n_actions)
    target_net = DQN_com(3*7*7, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
    memory = ReplayMemory(1000000, Transition)

    policy_net_com_free, reward_array_com_free = train_network_com(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, batch_size, eps_start, eps_end, gamma, Transition)

    torch.save(policy_net_com_free.state_dict(), "DQN_combined_500")

    plt.plot(reward_array_com_free)
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    policy_net_com_free.eval()

    env.reset()
    time = 0
    frames = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        observation = torch.tensor(observation).to(torch.float)
        observation = (torch.flatten(observation)).unsqueeze(0)
        if termination or truncation:
            a = None
        else:
            a = (policy_net_com_free(observation).max(1)[1]).item()
        env.step(a)
        frames.append(env.render())
        time+=1
    #     if time>=3000:
    #         break
    env.close()

    frames = [Image.fromarray(frame) for frame in frames]
    frame_one = frames[0]
    frame_one.save("DQN_combined_500.gif", format="GIF", append_images=frames,
                   save_all=True, duration=0.5, loop=0)

    batch_size = 32
    gamma = 1
    eps_start = 1
    eps_end = 0.1
    num_episodes = 500
    n_actions = 5

    env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=25,
    n_pursuers=6,obs_range=7, n_catch=2, freeze_evaders=True, tag_reward=0.01,
    catch_reward=5.0, urgency_reward=0, surround=True, constraint_window=1.0)

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    policy_net = DQN_com(3*7*7, n_actions)
    target_net = DQN_com(3*7*7, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
    memory = ReplayMemory(1000000, Transition)

    policy_net_com_frozen, reward_array_com_frozen = train_network_com(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, batch_size, eps_start, eps_end, gamma, Transition)

    torch.save(policy_net_com_frozen.state_dict(), "DQN_combined_500_frozen")

    plt.plot(reward_array_com_frozen)
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    policy_net_com_free.eval()

    env.reset()
    time = 0
    frames = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        observation = torch.tensor(observation).to(torch.float)
        observation = (torch.flatten(observation)).unsqueeze(0)
        if termination or truncation:
            a = None
        else:
            a = (policy_net_com_frozen(observation).max(1)[1]).item()
        env.step(a)
        frames.append(env.render())
        time+=1
    #     if time>=3000:
    #         break
    env.close()

    frames = [Image.fromarray(frame) for frame in frames]
    frame_one = frames[0]
    frame_one.save("DQN_combined_500.gif", format="GIF", append_images=frames,
                   save_all=True, duration=0.5, loop=0)


    batch_size = 32
    gamma = 1
    eps_start = 1
    eps_end = 0.1
    num_episodes = 300
    n_actions = 5

    env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=25,
    n_pursuers=6,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
    catch_reward=5.0, urgency_reward=0, surround=True, constraint_window=1.0)

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    policy_net = [DQN_sep(3*7*7, n_actions)  for _ in range(6)]
    target_net = [DQN_sep(3*7*7, n_actions)  for _ in range(6)]
    for i in range(6):
        target_net[i].load_state_dict(policy_net[i].state_dict())

    optimizer = [optim.RMSprop(policy_net[i].parameters(), lr=0.00025, alpha=0.95) for i in range(6)]
    memory = ReplayMemory(1000000, Transition)

    policy_net_sep, reward_array_sep = train_network(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, batch_size, eps_start, eps_end, gamma, Transition)

    plt.plot(reward_array_sep)
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    for i in range(6):
        torch.save(policy_net_sep[i].state_dict(),"DQN_separate"+str(i))

    env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=25,
    n_pursuers=6,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
    catch_reward=5.0, urgency_reward=0, surround=True, constraint_window=1.0, render_mode = 'rgb_array')

    policy_net = [DQN(3*7*7, 5)  for _ in range(6)]
    for i in range(6):
        policy_net[i].load_state_dict(torch.load("DQN_separate"+str(i)))
        policy_net[i].eval()

    env.reset()
    time = 0
    frames = []
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        observation = torch.tensor(observation).to(torch.float)
        observation = (torch.flatten(observation)).unsqueeze(0)
        if termination or truncation:
            a = None
        else:
            a = (policy_net[int(str(agent)[-1])](observation).max(1)[1]).item()
        env.step(a)
        frames.append(env.render())
        time+=1
    env.close()

    frames = [Image.fromarray(frame) for frame in frames]
    frame_one = frames[0]
    frame_one.save("DQN_separate.gif", format="GIF", append_images=frames,
                   save_all=True, duration=0.5, loop=0)

if __name__ == '__main__':
    main()
