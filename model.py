import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_ALPHABETA_MAX = 10
LOG_ALPHABETA_MIN = -10
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_hidden=2):
        super(ValueNetwork, self).__init__()

        self.input_layer = nn.Linear(num_inputs, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_hidden=2):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1_input_layer = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.q1_hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]
        self.q1_output_layer = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.q2_input_layer = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.q2_hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]
        self.q2_output_layer = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.q1_input_layer(xu))
        for layer in self.q1_hidden_layers:
            x1 = F.relu(layer(x1))
        x1 = self.q1_output_layer(x1)

        x2 = F.relu(self.q2_input_layer(xu))
        for layer in self.q2_hidden_layers:
            x2 = F.relu(layer(x2))
        x2 = self.q2_output_layer(x2)

        return x1, x2


class BetaPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, num_hidden=2):
        super(BetaPolicy, self).__init__()
        
        self.input_layer = nn.Linear(num_inputs, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]

        self.log_alpha_layer = nn.Linear(hidden_dim, num_actions)
        self.log_beta_layer = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(2.)
            self.action_bias = torch.tensor(-1.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) )
            self.action_bias = torch.FloatTensor(action_space.low)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        log_alpha = self.log_alpha_layer(x)
        log_beta = self.log_beta_layer(x)
        log_alpha = torch.clamp(log_alpha, min=LOG_ALPHABETA_MIN, max=LOG_ALPHABETA_MAX)
        log_beta = torch.clamp(log_beta, min=LOG_ALPHABETA_MIN, max=LOG_ALPHABETA_MAX)
        return log_alpha, log_beta

    def sample(self, state):
        log_alpha, log_beta = self.forward(state)
        alpha = log_alpha.exp() + 1.
        beta = log_beta.exp() + 1.
        beta = Beta(alpha, beta)
        x_t = beta.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = x_t * self.action_scale + self.action_bias
        log_prob = beta.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = beta.mean * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(BetaPolicy, self).to(device)
        

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, num_hidden=2):
        super(GaussianPolicy, self).__init__()
        
        self.input_layer = nn.Linear(num_inputs, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]

        self.mean_layer = nn.Linear(hidden_dim, num_actions)
        self.log_std_layer = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, num_hidden=2):
        super(DeterministicPolicy, self).__init__()
        self.input_layer = nn.Linear(num_inputs, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden-1)]

        self.mean_layer = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mean = torch.tanh(self.mean_layer(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
