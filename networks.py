import os
import torch as T
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


def weights_init_(m):
    # Initialize Policy weights
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)


# def initialize_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         T.nn.init.kaiming_uniform_(m.weight.data)
#         if m.bias is not None:
#             T.nn.init.constant_(m.bias.data, 0)


class CriticNetwork(nn.Module):
    def __init__(self, lr, state_space_dims, action_space_dims, fc1_dims=512, fc2_dims=512, name="critic", checkpoint_dir="./checkpoints"):
        super().__init__()

        # Model checkpoint info
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_sac.pth")

        # Model layer dimensions info
        self.state_space_dims = state_space_dims
        self.action_space_dims = action_space_dims
        self.input_dims = self.state_space_dims + self.action_space_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Model architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.apply(weights_init_)

        # Loss function
        # self.loss = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(), lr=lr
        )

        # Configure device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(device=self.device)

    def forward(self, state, action) -> T.Tensor:
        q_value = self.fc1(T.cat([state, action], dim=1))
        q_value = F.relu(q_value)
        q_value = self.fc2(q_value)
        q_value = F.relu(q_value)
        return self.q(q_value)


class ActorNetwork(nn.Module):
    def __init__(self, lr, state_space_dims, action_space_dims, max_action, fc1_dims=512, fc2_dims=512, name="actor", checkpoint_dir="./checkpoints"):
        super(ActorNetwork, self).__init__()

        # Model checkpoint info
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + "_sac.pth")

        # Model layer dimensions info
        self.input_dims = state_space_dims
        self.action_space_dims = action_space_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # Max action in action space
        self.max_action = max_action

        # Reparameterization
        self.reparam_noise = 1e-6
        self.log_sigma_min = -20
        self.log_sigma_max = 2

        # Model architecture
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.action_space_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.action_space_dims)

        self.apply(weights_init_)

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(), lr=lr
        )

        # Configure device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(device=self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)

        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.log_sigma_min, max=self.log_sigma_max)

        return mu, sigma

    def sample_normal(self, state: T.Tensor, reparameterize=True):
        mu, sigma = self.forward(state)

        if reparameterize:
            std = sigma.exp()
            probabilities = Normal(mu, std)
            sampled_actions = probabilities.rsample()

            y_t = T.tanh(sampled_actions)
            scaled_actions = y_t * self.max_action

            log_probs: T.Tensor = probabilities.log_prob(sampled_actions)
            log_probs -= T.log(self.max_action *
                               (1 - y_t.pow(2)) + self.reparam_noise)
            log_probs = log_probs.sum(1, keepdim=True)
        else:
            scaled_actions = T.tanh(mu)
            log_probs = None

        return scaled_actions, log_probs
