import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, ValueNetwork, DeterministicPolicy, BetaPolicy


class SAC(object):
    def __init__(self, 
        num_inputs,
        action_space,
        policy,
        gamma,
        tau,
        lr,
        alpha,
        automatic_entropy_tuning,
        hidden_size,
        target_update_interval,
        cuda,
        use_value_function,
        eps=1e-8,
        ):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if cuda else "cpu")
        if not torch.cuda.is_available():
            self.device = "cpu"

        self.use_value_function = use_value_function

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=eps)

        if self.use_value_function:
            self.value = ValueNetwork(num_inputs, hidden_size).to(self.device)
            self.value_target = ValueNetwork(num_inputs, hidden_size).to(self.device)
            self.value_optim = Adam(self.value.parameters(), lr=lr, eps=eps)
            hard_update(self.value_target, self.value)
        else:
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
            hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr, eps=eps)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=eps)

        elif self.policy_type == "Deterministic":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=eps)

        elif self.policy_type == "Beta":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr, eps=eps)

            self.policy = BetaPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr, eps=eps)

        else:
            raise ValueError('Invalid policy type %s'%(self.policy_type))

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            if self.use_value_function:
                vf_next_target = self.value_target(next_state_batch)
                next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)
            else:
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.use_value_function:
            with torch.no_grad():
                vf_target = min_qf_pi - (self.alpha * log_pi)
            vf = self.value(state_batch)
            vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

            self.value_optim.zero_grad()
            vf_loss.backward()
            self.value_optim.step()
        else:
            vf_loss = torch.tensor(0.).to(self.device)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            if self.use_value_function:
                soft_update(self.value_target, self.value, self.tau)
            else:
                soft_update(self.critic_target, self.critic, self.tau)

        return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        if self.use_value_function:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'value_state_dict': self.value.state_dict(),
                        'value_target_state_dict': self.value_target.state_dict(),
                        'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                        'value_optimizer_state_dict': self.value_optim.state_dict(),
                        'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)
        else:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'critic_target_state_dict': self.critic_target.state_dict(),
                        'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                        'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            if self.use_value_function:
                self.value.load_state_dict(checkpoint['value_state_dict'])
                self.value_target.load_state_dict(checkpoint['value_target_state_dict'])
                self.value_optim.load_state_dict(checkpoint['value_optimizer_state_dict'])
            else:
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                if self.use_value_function:
                    self.value.eval()
                    self.value_target.eval()
                else:
                    self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                if self.use_value_function:
                    self.value.train()
                    self.value_target.train()
                else:
                    self.critic_target.train()

