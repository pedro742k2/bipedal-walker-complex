from random import random
import numpy as np
from networks import ActorNetwork, CriticNetwork
import torch as T
from replay_memory import ReplayMemory
from torch import nn
from utils import get_date_time


class Agent:
    def __init__(self, start_steps, gamma, max_replay_size, tau, alpha, batch_size, actor_lr, critic_lr, state_space_dims, action_space_dims, max_action, min_action, runtest_folder_name):
        # Configure device,
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        if "cuda" not in self.device.type:
            print("[WARNING] USING CPU INSTEAD OF GPU")
        else:
            print("[INFO] USING CUDA")

        # Environment config
        self.state_space_dims = state_space_dims
        self.action_space_dims = action_space_dims
        self.max_action = max_action
        self.min_action = min_action

        # Hyperparameters config
        self.gamma = gamma
        self.tau = tau

        self.target_entropy = - \
            T.prod(T.tensor(self.action_space_dims,
                   dtype=T.float32, device=self.device)).item()

        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=actor_lr)
        self.alpha = T.tensor([alpha], dtype=T.float32, device=self.device)
        # self.target_entropy = -4
        # self.entropy_alpha = entropy_alpha

        self.runtest_folder_name = runtest_folder_name

        self.log_file_alpha = open(
            f"{self.runtest_folder_name}/logs/alpha.log", "w")

        self.start_steps = start_steps
        self.step_counter = 0
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Replay memory config
        self.replay_memory = ReplayMemory(
            max_size=max_replay_size, input_shape=(self.state_space_dims,), n_actions=self.action_space_dims)
        self.batch_size = batch_size

        # Configure online networks
        self.actor = ActorNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.actor_lr,
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims,
            max_action=self.max_action)
        self.critic_1 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr,
            name="critic_1",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)
        self.critic_2 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr, name="critic_2",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)
        self.critic_3 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr, name="critic_3",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)

        # Configure target networks
        self.target_critic_1 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr,
            name="target_critic_1",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)
        self.target_critic_2 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr, name="target_critic_2",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)
        self.target_critic_3 = CriticNetwork(
            checkpoint_dir=f"{runtest_folder_name}/checkpoints",
            lr=self.critic_lr, name="target_critic_3",
            state_space_dims=self.state_space_dims,
            action_space_dims=self.action_space_dims)

        # Configure alpha network
        # self.alpha_network = AlphaNetwork(lr=self.actor_lr)

        # MSE loss function
        self.mse_loss = nn.MSELoss()

        self.test_log_file = open(
            f"{runtest_folder_name}/logs/losses.log", "w")
        # TODO: Remove this
        self.alpha_values = []
        self.alpha_loss = []

        self.update_target_parameters(tau=1)

    def update_target_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_critic_1_param, critic_1_param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_critic_1_param.data.copy_(
                target_critic_1_param.data *
                (1.0-tau) + critic_1_param.data * tau
            )

        for target_critic_2_param, critic_2_param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_critic_2_param.data.copy_(
                target_critic_2_param.data *
                (1.0-tau) + critic_2_param.data * tau
            )

        for target_critic_3_param, critic_3_param in zip(self.target_critic_3.parameters(), self.critic_3.parameters()):
            target_critic_3_param.data.copy_(
                target_critic_3_param.data *
                (1.0-tau) + critic_3_param.data * tau
            )

    def remember(self, state, action, reward, new_state, done):
        self.replay_memory.store_transition(
            state, action, reward, new_state, done)

    def save_models(self):
        T.save(self.actor.state_dict(), self.actor.checkpoint_file)
        T.save(self.critic_1.state_dict(), self.critic_1.checkpoint_file)
        T.save(self.critic_2.state_dict(), self.critic_2.checkpoint_file)
        T.save(self.critic_3.state_dict(), self.critic_3.checkpoint_file)

    def load_models(self):
        try:
            self.actor.load_state_dict(
                T.load(self.actor.checkpoint_file))
            self.critic_1.load_state_dict(
                T.load(self.critic_1.checkpoint_file))
            self.critic_2.load_state_dict(
                T.load(self.critic_2.checkpoint_file))
            self.critic_3.load_state_dict(
                T.load(self.critic_3.checkpoint_file))
            print("Models loaded successfully.")
        except:
            print("Something went wrong loading the models.")

    def select_actions(self, state, add_noise):
        if self.step_counter < self.start_steps:
            return [random()*self.max_action for _ in range(self.action_space_dims)]
        else:
            self.actor.eval()  # Sets the actor in evaluation mode

            state_tensor = T.tensor(
                np.array([state]), dtype=T.float32, device=self.device)
            actions, _ = self.actor.sample_normal(
                state_tensor, reparameterize=add_noise)

            self.actor.train()  # Sets the actor in training mode

            return actions.detach().cpu().numpy()[0]

    def learn(self):
        if self.replay_memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.replay_memory.sample_buffer(
            self.batch_size)

        state_batch = T.tensor(state, dtype=T.float32, device=self.device)
        action_batch = T.tensor(action, dtype=T.float32, device=self.device)
        reward_batch = T.tensor(reward, dtype=T.float32, device=self.device)
        new_state_batch = T.tensor(
            new_state, dtype=T.float32, device=self.device)
        done_batch = T.tensor(done, dtype=T.float32, device=self.device)

        with T.no_grad():
            # Sample new actions and respective log prob.
            new_actions, new_action_logs = self.actor.sample_normal(
                new_state_batch, reparameterize=True)

            next_q1_target = self.target_critic_1.forward(
                new_state_batch, new_actions)
            next_q2_target = self.target_critic_2.forward(
                new_state_batch, new_actions)
            next_q3_target = self.target_critic_3.forward(
                new_state_batch, new_actions)

            next_q_target = T.cat(
                (next_q1_target, next_q2_target, next_q3_target), dim=1)

            q_next = T.min(next_q_target, 1).values - \
                self.alpha * new_action_logs.squeeze(1)

            # Calculate targets
            targets = reward_batch + self.gamma * (1-done_batch) * q_next

        # Get critic values
        q1_values = self.critic_1.forward(state_batch, action_batch).squeeze(1)
        q2_values = self.critic_2.forward(state_batch, action_batch).squeeze(1)
        q3_values = self.critic_3.forward(state_batch, action_batch).squeeze(1)

        # Update critic 1
        q1_values = self.critic_1.forward(state_batch, action_batch).squeeze(1)
        self.critic_1.optimizer.zero_grad()
        q1_loss = self.mse_loss.forward(q1_values, targets)
        q1_loss.backward()
        self.critic_1.optimizer.step()

        # Update critic 2
        q2_values = self.critic_2.forward(state_batch, action_batch).squeeze(1)
        self.critic_2.optimizer.zero_grad()
        q2_loss = self.mse_loss.forward(q2_values, targets)
        q2_loss.backward()
        self.critic_2.optimizer.step()

        # Update critic 3
        q3_values = self.critic_3.forward(state_batch, action_batch).squeeze(1)
        self.critic_3.optimizer.zero_grad()
        q3_loss = self.mse_loss.forward(q3_values, targets)
        q3_loss.backward()
        self.critic_3.optimizer.step()

        with T.no_grad():
            self.test_log_file.write(
                f"\nReward batch mean: {reward_batch.mean()}\n")
            self.test_log_file.write(f"TARGET Q: {targets.mean()}\n")
            self.test_log_file.write(f"Q1 value: {q1_values.mean()}\n")
            self.test_log_file.write(f"Q1 loss: {q1_loss}\n")

        # Update actor network
        if self.step_counter % 2 == 0:
            # Sample new actions and respective log prob. with reparam.
            policy_actions, policy_action_logs = self.actor.sample_normal(
                state_batch, reparameterize=True)

            q1 = self.critic_1.forward(state_batch, policy_actions)
            q2 = self.critic_2.forward(state_batch, policy_actions)
            q3 = self.critic_3.forward(state_batch, policy_actions)

            cat_q = T.cat((q1, q2, q3), dim=1)

            q_values = T.min(cat_q, 1).values

            # Get policy loss
            policy_loss = (self.alpha *
                           policy_action_logs - q_values).mean()

            with T.no_grad():
                self.test_log_file.write(
                    f"ACTOR LOSS: {policy_loss.mean()}\n")
                self.test_log_file.write("-"*50)

            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

        # Update alpha network
        _, policy_action_logs = self.actor.sample_normal(
            state_batch, reparameterize=True)

        alpha_loss = -(self.log_alpha * (policy_action_logs +
                       self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()       # For logs
        alpha_tlogs = self.alpha.clone()        # For logs
        self.alpha_values.append(alpha_tlogs)   # For logs

        self.log_file_alpha.write(str(alpha_tlogs) + "\n")

        self.update_target_parameters()

        self.step_counter += 1
