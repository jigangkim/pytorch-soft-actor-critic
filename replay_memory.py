import random
import numpy as np
import os
import pickle

from buffer import HindsightReplayBuffer as HER_backend

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, episode_num=None, info_dict=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

class HindsightReplayMemory:
    def __init__(self, capacity, seed, env, n_sampled_goal=4, goal_selection_strategy='future', online_sampling: bool=True):
        random.seed(seed)
        self.capacity = capacity
        self.env = env
        self.backend = HER_backend(
            capacity,
            env.compute_reward,
            n_sampled_goal,
            goal_selection_strategy,
            online_sampling
        )

    def push(self, state, action, reward, next_state, done, episode_num, info_dict):
        transition = {}
        state_dict = self.env.get_obs_dict(state)
        transition['observation'] = state_dict['observation']
        transition['achieved_goal'] = state_dict['achieved_goal']
        transition['desired_goal'] = state_dict['desired_goal']
        transition['action'] = action
        transition['reward'] = reward
        next_state_dict = self.env.get_obs_dict(next_state)
        transition['next_observation'] = next_state_dict['observation']
        transition['next_achieved_goal'] = next_state_dict['achieved_goal']
        transition['next_desired_goal'] = next_state_dict['desired_goal']
        transition['done'] = True # done always false for HER & 'done' used as mask(=1 - done) here
        transition['episode'] = episode_num
        transition['info'] = info_dict
        self.backend.store(transition)

    def sample(self, batch_size):
        batch = self.backend.sample(batch_size)
        batch_states = np.concatenate([batch['observations'], batch['achieved_goals'], batch['desired_goals']], axis=-1)
        batch_next_states = np.concatenate([batch['next_observations'], batch['next_achieved_goals'], batch['next_desired_goals']], axis=-1)
        return batch_states, batch['actions'], np.squeeze(batch['rewards']), batch_next_states, np.squeeze(batch['dones'])

    def __len__(self):
        return self.backend.num_transitions