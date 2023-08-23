import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from jax import random
from multigroupGP import RBF, MultiGroupRBF, GP 
from sklearn.metrics import mean_squared_error


class Agent:
    def __init__(self, nfq_net: nn.Module, optimizer: optim.Optimizer, random_seed=24, dir="models", mode="mggp"):
        self.kernel = MultiGroupRBF()
        self.RANDOM_SEED = random_seed
        self.key = random.PRNGKey(self.RANDOM_SEED)
        self.gp_q_est = GP(self.kernel, key = self.key, is_mggp=True)
        self.nfq_net = nfq_net
        self.optimizer = optimizer
        self.dir = dir
        self.mode = mode
        self.unique_actions = None

    def get_best_action(self, obs: np.array, group) -> int:
        if self.mode == "mggp":
            q_list = np.zeros(len(self.unique_actions))
            obs =jnp.array(obs)

            for ii, a in enumerate(self.unique_actions):
                x = jnp.append(obs, a)
                x = x.reshape(1,-1)
                q = self.gp_q_est.predict(x, group)
                q_list[ii] = q
            return self.unique_actions[np.argmin(q_list)]
        else:
            q_list = np.zeros(len(self.unique_actions))
            for ii, a in enumerate(self.unique_actions):
                q = self.nfq_net(
                torch.cat([torch.FloatTensor(obs), torch.FloatTensor(a)], dim=0))
                q_list[ii] = q
            return self.unique_actions[np.argmin(q_list)]
    
    def train(self, pattern_set: tuple) -> float:
        if self.mode == "nfq":
            state_action_b, target_q_values, group_b = pattern_set
            predicted_q_values = self.nfq_net(state_action_b, group_b).squeeze()
            loss = torch.nn.functional.mse_loss(predicted_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss       
        else:
            X, y, group_b = pattern_set
            X = jnp.array(X)
            y = jnp.array(y)
            group_b = jnp.array(group_b)
            x_train, x_test, y_train, y_test, g_train, g_test = train_test_split(X, y, group_b, test_size=0.95) 
            x_test, _, y_test, _, g_test, _ = train_test_split(x_test, y_test, g_test, test_size=0.95)
            self.gp_q_est.fit(x_train, y_train, g_train)
            return mean_squared_error(self.gp_q_est.predict(x_test, g_test), y_test)

    def save(self):
        if self.mode == "mggp":
            with open(self.dir + "mggp_model.dump", "wb") as f:
                pickle.dump(self.gp_q_est, f)
        else:
            torch.save(self.nfq_net.state_dict(), self.dir + "nfq_net.pth")

    def load(self):
        if self.mode == "mggp":
            self.gp_q_est = pickle.load(open(self.dir + "mggp_model.dump", "rb"))
        else:
            self.nfq_net.load_state_dict(torch.load(self.dir + "nfq_net.pth"))

    def generate_pattern_set(self, rollouts, gamma: float = 0.95):
        state_b, action_b, next_state_b, reward_b, done_b, group_b = rollouts
        target_q_values = self.get_target_q_values(reward_b,next_state_b, done_b, group_b, gamma)
        q_values = self.get_q_values(state_b, action_b, group_b)
        state_action_b = self.get_state_action_pairs(state_b, action_b)
        return state_action_b, target_q_values, group_b

    def get_target_q_values(self, reward_b, next_state_b, done_b, group, gamma):
        target_q_values = []
        for r, next_state, done in zip(reward_b, next_state_b, done_b):
            if done:
                target_q_values.append(r)
            else:
                best_action = self.get_best_action(next_state, group)
                target_q = r + gamma * self.get_q_values(next_state, best_action, group)
                target_q_values.append(target_q)
        return np.array(target_q_values)

    def get_q_values(self, state_b, action_b, group_b):
        q_values = []
        for state, action, group in zip(state_b, action_b, group_b):
            q_values.append(self.get_best_action(state, group))
        return np.array(q_values)

    def get_state_action_pairs(self, state_b, action_b):
        state_action_pairs = []
        for state, action in zip(state_b, action_b):
            state_action_pairs.append(np.concatenate((state, action)))
        return np.array(state_action_pairs)


