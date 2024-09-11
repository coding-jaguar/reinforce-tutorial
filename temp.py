import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=-1)

model = NeuralNetwork()
optimizer = Adam(model.parameters(), lr=0.001)

env = gym.make('CartPole-v1')


num_episodes = 1000
episode_returns = []

for epoc in range(num_episodes):
    log_probs = []
    rewards = []
    state = torch.tensor(env.reset()[0], dtype=torch.float32)
    while True:
        action_probs = Categorical(model(state))
        action = action_probs.sample()
        state, rew, terminated, trunc, info = env.step(action.item())
        state = torch.tensor(state, dtype=torch.float32)
        rewards.append(rew)
        log_probs.append(action_probs.log_prob(action))

        if terminated:
            break

    returns = [None]* len(rewards)
    s = 0
    baseline = sum(rewards)/len(rewards)

    for t in range(len(rewards)-1,-1,-1):
        s += rewards[t]
        returns[t] = s - baseline

    log_probs = torch.stack(log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)

    loss =  - log_probs * returns
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(returns[0])
    episode_returns.append(returns[0].item())

    if returns[0].item()>5000:
        torch.save(model, "solution_network.pth")
        break

plt.plot(episode_returns)
plt.show()

env = gym.make("CartPole-v1", render_mode="human")
model = torch.load("solution_network.pth")
state = env.reset()[0]
while True:
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = model(state)
    action = torch.argmax(action_probs)
    state,rew,terminated,trunc,_ = env.step(action.item())
    if terminated: break

