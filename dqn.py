import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Try to import gymnasium first (preferred), fall back to gym
try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        GYM_VERSION = "gym"
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is installed. Please install one of them.")


class DQN(nn.Module):
    """Deep Q-Network for CartPole environment."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # store as numpy arrays for convenience
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state, online_net, epsilon, action_dim, device):
    """ε-greedy action selection."""
    if random.random() < epsilon:
        # explore
        return random.randint(0, action_dim - 1)
    else:
        # exploit
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = online_net(state_t)  # shape [1, action_dim]
        action = q_values.argmax(dim=1).item()
        return action


def train_step(
    online_net,
    target_net,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device
):
    """One training step with Bellman update."""
    if len(replay_buffer) < batch_size:
        return None  # not enough data yet
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)  # [B,1]
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # Q(s,a) for the actions actually taken
    q_values = online_net(states_t)                  # [B, action_dim]
    q_sa = q_values.gather(1, actions_t)             # [B,1]

    # Compute target: r + γ * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q_values = target_net(next_states_t)    # [B, action_dim]
        max_next_q = next_q_values.max(dim=1, keepdim=True)[0]  # [B,1]
        target = rewards_t + gamma * max_next_q * (1.0 - dones_t)

    # Loss = MSE(target, q_sa)
    loss = nn.MSELoss()(q_sa, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def reset_env(env):
    """Handle env.reset() for both gym and gymnasium."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        # gymnasium or newer gym: (observation, info)
        state, info = reset_result
        return state, info
    else:
        # older gym: just observation
        return reset_result, {}


def step_env(env, action):
    """Handle env.step() for both gym and gymnasium."""
    step_result = env.step(action)
    
    if len(step_result) == 5:
        # gymnasium or newer gym: (observation, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return next_state, reward, done, terminated, truncated, info
    elif len(step_result) == 4:
        # older gym: (observation, reward, done, info)
        next_state, reward, done, info = step_result
        terminated = done
        truncated = False
        return next_state, reward, done, terminated, truncated, info
    else:
        # Very old gym: (observation, reward, done)
        next_state, reward, done = step_result
        terminated = done
        truncated = False
        info = {}
        return next_state, reward, done, terminated, truncated, info


def train_dqn(
    env_name="CartPole-v1",
    max_episodes=500,
    gamma=0.99,
    batch_size=64,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    target_update_freq=1000,
    replay_buffer_capacity=50000,
    learning_rate=1e-3,
    device=None
):
    """Full DQN training loop."""
    
    # Setup environment
    env = gym.make(env_name)
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {GYM_VERSION} for environment")

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Initialize networks
    online_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())  # copy weights
    target_net.eval()  # we don't train target directly

    # Setup optimizer and replay buffer
    optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    # Training parameters
    epsilon = epsilon_start
    global_step = 0
    
    # Training statistics
    episode_rewards = []
    
    print("\nStarting DQN training...")
    print(f"Max episodes: {max_episodes}")
    print(f"Gamma: {gamma}, Batch size: {batch_size}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_min} (decay: {epsilon_decay})")
    print(f"Target update frequency: {target_update_freq} steps\n")

    for episode in range(max_episodes):
        # Reset environment (handle both gym and gymnasium)
        state, _ = reset_env(env)
        
        episode_reward = 0
        episode_losses = []
        done = False

        while not done:
            global_step += 1

            # 1. Select action
            action = select_action(state, online_net, epsilon, action_dim, device)

            # 2. Step the environment (handle both gym and gymnasium)
            next_state, reward, done, terminated, truncated, info = step_env(env, action)

            # 3. Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # 4. Train step
            loss = train_step(
                online_net, target_net, optimizer,
                replay_buffer, batch_size, gamma, device
            )
            if loss is not None:
                episode_losses.append(loss)

            # 5. Update target network
            if global_step % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

        # 6. Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_reward

        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode+1:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg Reward (last 100): {avg_reward:6.1f} | "
                  f"Epsilon: {epsilon:.3f} | Loss: {avg_loss:.4f}")

    env.close()
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    return online_net, target_net, episode_rewards


if __name__ == "__main__":
    # Train the DQN
    online_net, target_net, rewards = train_dqn(
        env_name="CartPole-v1",
        max_episodes=500,
        gamma=0.99,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_freq=1000,
        replay_buffer_capacity=50000,
        learning_rate=1e-3
    )

