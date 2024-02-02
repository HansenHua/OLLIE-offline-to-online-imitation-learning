from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F

TensorBatch = List[torch.Tensor]
EPS = 1e-7

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(500)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # otof
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks

    # AntMaze hacks
    bc_steps: int = int(1000)  # Number of BC steps at start
    reward_scale: float = 5.0
    reward_bias: float = -1.0
    policy_log_std_multiplier: float = 1.0

    # offline RL
    use_offline : bool = False
    alpha: float = 0.5


    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def extract_actor_param(agent):
	agent_param = []
	for group in agent.actor_optimizer.param_groups:
		for p in group['params']:
			if p is None:
				agent_param.append(torch.zeros_like(p).to(p.device))
				continue
			# print(f"learner 1 {p.grad[0]}")
			agent_param.append(p.clone())
	return agent_param

def flatten_param(param):
    flatten_param = torch.cat([g.flatten() for g in param])
    return  flatten_param

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env



def dataset_split_diy(dataset, size, num_dataset, terminate_on_end=False):
    n = dataset['rewards'].shape[0]
    print(n)
    split_point = size
    # return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]
    timeout_traj = [[]]

    for i in range(n - 1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i + 1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))
        # timeout_traj[-1].append(bool(dataset['timeouts'][i]))

        # final_timestep = dataset['timeouts'][i] | dataset['terminals'][i]
        if (not terminate_on_end) and i != n - 2:
            # Skip this transition and don't apply terminals on the last step of an episode
            # return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])
            timeout_traj.append([])

    # Select trajs
    inds_data = []
    inds_all = list(range(len(obs_traj)))
    for i in range (num_dataset):
        if i == 0:
            inds_data.append(inds_all[:split_point])
        else:
            inds_data.append(inds_all[i * split_point: (i + 1) * split_point])


    # inds_data_2 = list(inds_data_2)
    # inds_data_1 = list(inds_data_1)

    # print('# {} expert trajs in D_1'.format(len(inds_data_1)))
    # print('# {} expert trajs in D_2'.format(len(inds_data_2)))

    obs_traj_subset = []
    next_obs_subset = []
    action_traj_subset = []
    reward_traj_subset = []
    done_traj_subset = []
    for i in range(num_dataset):
        obs_traj_subset.append([])
        next_obs_subset.append([])
        action_traj_subset.append([])
        reward_traj_subset.append([])
        done_traj_subset.append([])
    for i in range(num_dataset):
        obs_traj_subset[i] = [obs_traj[j] for j in list(inds_data[i])]
        next_obs_subset[i] = [next_obs_traj[j] for j in list(inds_data[i])]
        action_traj_subset[i] = [action_traj[j] for j in list(inds_data[i])]
        reward_traj_subset[i] = [reward_traj[j] for j in list(inds_data[i])]
        done_traj_subset[i] = [done_traj[j] for j in list(inds_data[i])]


    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    dataset_stack = []
    for i in range(num_dataset):
        dataset_dict = {
            'observations': concat_trajectories(obs_traj_subset[i]),
            'actions': concat_trajectories(action_traj_subset[i]),
            'next_observations': concat_trajectories(next_obs_subset[i]),
            'rewards': concat_trajectories(reward_traj_subset[i]),
            'terminals': concat_trajectories(done_traj_subset[i]),
            # 'timeouts': concat_trajectories(timeout_traj_1),
        }
        dataset_stack.append(dataset_dict)




    return dataset_stack



class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._mc_returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            self._states = torch.zeros((self._buffer_size, self.state_dim), dtype=torch.float32, device=self._device)
            self._actions = torch.zeros((self._buffer_size, self.action_dim), dtype=torch.float32, device=self._device)
            self._rewards = torch.zeros((self._buffer_size, 1), dtype=torch.float32, device=self._device)
            self._mc_returns = torch.zeros((self._buffer_size, 1), dtype=torch.float32, device=self._device)
            self._next_states = torch.zeros((self._buffer_size, self.state_dim), dtype=torch.float32, device=self._device)
            self._dones = torch.zeros((self._buffer_size, 1), dtype=torch.float32, device=self._device)
            self._pointer = 0
            self._size = 0
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._mc_returns[:n_transitions] = self._to_tensor(data.get("mc_returns")[..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(self._pointer)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        mc_returns = self._mc_returns[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, mc_returns, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    # actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob

class TanhGaussianDistribution(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return actions, log_probs

    def get_log_density(self, observations, action):
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.tanh_gaussian.log_std_min, self.tanh_gaussian.log_std_max)
        std = torch.exp(log_std)
        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_prob = torch.sum(action_distribution.log_prob(action_clip), dim=-1)

        return logp_prob
    
    def train(self, dataset, batch_size):
        for i in range(self.config.bc_steps):
            state_e, action_e, _, _, _, _, _, _ = dataset.sample(batch_size)
            loss = - self.get_log_density(state_e, action_e).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    def get_log_density(self, observations, action):
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.tanh_gaussian.log_std_min, self.tanh_gaussian.log_std_max)
        std = torch.exp(log_std)
        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_prob = torch.sum(action_distribution.log_prob(action_clip), dim=-1)

        return logp_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class Trainer(nn.Module):
    def __init__(self, config, actor, critic, y_function, policy, distribution):
        self.config = config
        self.actor = actor
        self.critic = critic
        self.policy = policy
        self.lamda = Scalar(1)
        self.y_function = y_function
        self.distribution = distribution
        self.lamda_optimizer = torch.optim.Adam(self.lamda.parameters(), lr=config.qf_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.qf_lr)
        self.y_optimizer = torch.optim.Adam(self.y_function.parameters(), lr=config.qf_lr)

    def train_critic(self, observations, actions, next_observations, rewards, dones, alpha):
        action_prime, _ = self.actor(next_observations).detach()
        adv = rewards + self.config.discount * (1 - dones) * self.critic(next_observations, action_prime) - self.critic(observations, actions)
        y = self.y_function(observations, actions).detach()
        loss = alpha * torch.mean(adv * y - alpha * self.lamda.detach() * y * torch.log(alpha * y))
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def train_reward(self, observations, actions, next_observations, rewards, dones, alpha):
        action_prime, _ = self.actor(next_observations).detach()
        adv = rewards + self.config.discount * (1 - dones) * self.critic(next_observations, action_prime).detach() - self.critic(observations, actions).detach()
        y = self.y_function(observations, actions)
        loss = - alpha * torch.mean(adv * y - alpha * self.lamda.detach() * y * torch.log(alpha * y))
        self.y_optimizer.zero_grad()
        loss.backward()
        self.y_optimizer.step()
    
    def train_lamda(self, observations, actions, next_observations, rewards, dones, alpha):
        action_prime, _ = self.actor(next_observations).detach()
        adv = rewards + self.config.discount * (1 - dones) * self.critic(next_observations, action_prime).detach() - self.critic(observations, actions).detach()
        y = self.y_function(observations, actions).detach()
        loss = torch.mean(adv * y - alpha * self.lamda * y * torch.log(alpha * y))
        self.lamda_optimizer.zero_grad()
        loss.backward()
        self.lamda_optimizer.step()

    def train_policy(self, batch, use_sac =False, use_norm=True):
        observations, actions, rewards, next_observations, dones = batch
        if use_sac:
            log_probs = self.actor.get_log_density(observations, actions)
            y = self.y_function(observations, actions).detach()
            d = self.policy.discriminator(observations, actions).detach()
            phi = self.distribution.get_log_density(observations, actions)
            x = phi * y * (1 / d - 1)
            loss = log_probs.mean() - torch.mean(torch.log(x))
        else:
            if use_norm:
                log_probs = self.actor.get_log_density(observations, actions)
                y = self.y_function(observations, actions).detach()
                y_mean = y.mean()
                loss = y / y_mean * log_probs
            else:
                log_probs = self.actor.get_log_density(observations, actions)
                y = self.y_function(observations, actions).detach()
                loss = y * log_probs
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def train_online(self, batch, expert_data):
        observations, actions, rewards, mc_returs, next_observations, dones = batch
        exp_observations, exp_actions, exp_rewards, exp_next_observations, exp_dones = batch
        log_probs = self.actor.log_prob(observations, actions)
        loss = log_probs.mean() - torch.mean(torch.log(log_probs * mc_returs))
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.train_disc(observations, actions, exp_observations, exp_actions)
    
    def train_disc(self, states, actions, states_exp, actions_exp):
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.y_optimizer.zero_grad()
        loss_disc.backward()
        self.y_optimizer.step()

    def train_offline(self, batch, alpha, non_discount=False):
        observations, actions, rewards, next_observations, dones = batch
        self.train_critic(observations, actions, next_observations, rewards, dones, alpha)
        if non_discount:
            self.train_lamda(observations, actions, next_observations, rewards, dones, alpha)
        self.train_reward(observations, actions, next_observations, rewards, dones, alpha)
        # self.train_policy(observations, actions, next_observations, rewards, dones, alpha)

class OTOF(nn.Module):
    def __init__(self, env, state_dim, action_dim, max_action, dataset, config, policy, distribution):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.dataset = dataset
        self.config = config
        self.policy = policy
        self.distribution = distribution
        self.replay_buffer = ReplayBuffer(
			state_dim,
			action_dim,
			config.buffer_size,
			config.device,
		)

        self.actor = TanhGaussianPolicy(
			state_dim,
			action_dim,
			max_action,
			log_std_multiplier=config.policy_log_std_multiplier,
			orthogonal_init=config.orthogonal_init,
		).to(config.device)

        self.critic = FullyConnectedQFunction(
			state_dim,
			action_dim,
			config.orthogonal_init
		).to(config.device)

        self.y_function = FullyConnectedQFunction(
			state_dim,
			action_dim,
			config.orthogonal_init
		).to(config.device)

        self.trainer = Trainer(self.config, self.actor, self.critic, self.y_function, self.policy, self.distribution)