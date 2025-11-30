# Double DQN Implementation for CartPole

A complete **Double Deep Q-Network (Double DQN)** implementation for solving the CartPole-v1 environment using PyTorch. This implementation follows the original DQN paper by Mnih et al. (2015) with improvements from van Hasselt et al. (2016) for Double DQN.

## 🚀 Features

- **Double DQN**: Reduces Q-value overestimation bias compared to standard DQN
- **Experience Replay**: Replay buffer for stable training and data efficiency
- **Target Network**: Separate target network for stable Q-learning updates
- **ε-greedy Exploration**: Decaying epsilon for exploration-exploitation balance
- **Evaluation Function**: Greedy policy evaluation during training
- **Early Stopping**: Automatically stops when the environment is solved (avg reward ≥ 195)
- **Reproducibility**: Seeded random number generators for reproducible results
- **Compatible**: Works with both `gym` and `gymnasium` (auto-detects)
- **GPU Support**: Automatically uses CUDA if available

## 📋 Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy
- Gym or Gymnasium

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/hamidreza23/DQN-CartPole.git
cd DQN-CartPole

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Training

Simply run the training script:

```bash
python dqn.py
```

### Custom Training Parameters

You can modify the training parameters in `dqn.py` or import and use the `train_dqn` function:

```python
from dqn import train_dqn

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
```

### Evaluate a Trained Policy

```python
from dqn import evaluate_policy, reset_env, step_env
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Evaluate greedy policy (no exploration)
avg_reward = evaluate_policy(env, online_net, device, n_episodes=10)
print(f"Average reward: {avg_reward}")
```

## 🏗️ Architecture

### Environment
- **State Space**: 4 dimensions (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (push left, push right)

### Neural Network
```
Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Output (2)
```

The network outputs Q-values for each action, representing the expected future reward.

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Gamma (γ)** | 0.99 | Discount factor for future rewards |
| **Batch Size** | 64 | Number of samples per training step |
| **Learning Rate** | 1e-3 | Adam optimizer learning rate |
| **Epsilon Start** | 1.0 | Initial exploration rate (100% random) |
| **Epsilon Min** | 0.05 | Minimum exploration rate (5% random) |
| **Epsilon Decay** | 0.995 | Exploration decay per episode |
| **Target Update Frequency** | 1000 steps | How often to sync target network |
| **Replay Buffer Capacity** | 50,000 | Maximum stored transitions |
| **Seed** | 0 | Random seed for reproducibility |

## 🔑 Key Components

1. **DQN Class**: Neural network for Q-value approximation
2. **ReplayBuffer**: Stores and samples experiences (state, action, reward, next_state, done)
3. **select_action**: ε-greedy action selection (exploration vs exploitation)
4. **train_step**: Double DQN Bellman update step
5. **evaluate_policy**: Greedy policy evaluation (epsilon=0)
6. **train_dqn**: Complete training loop with statistics tracking

## 🧠 Double DQN vs Standard DQN

Standard DQN tends to overestimate Q-values because it uses the same network to select and evaluate actions. Double DQN addresses this by:

1. **Online network** selects the best action: `a* = argmax_a Q_online(s', a)`
2. **Target network** evaluates that action: `Q_target(s', a*)`

```python
# Double DQN target computation
best_next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
max_next_q = target_net(next_states).gather(1, best_next_actions)
target = rewards + gamma * max_next_q * (1 - dones)
```

## 📊 Training Output

The training shows progress every 10 episodes with evaluation every 20 episodes:

```
Episode  440 | Reward:   78.0 | Avg Reward (last 100):  163.3 | Epsilon: 0.110 | Loss: 0.7211 | Eval Reward:  500.0
Episode  450 | Reward:  335.0 | Avg Reward (last 100):  164.7 | Epsilon: 0.105 | Loss: 0.5640
Episode  460 | Reward:  104.0 | Avg Reward (last 100):  172.2 | Epsilon: 0.100 | Loss: 1.0488 | Eval Reward:  209.8
```

## 📈 Expected Performance

- The agent typically reaches **evaluation rewards of 500** (perfect score) within 400-450 episodes
- Final average reward: ~180+ over last 100 episodes
- Early stopping triggers when avg reward ≥ 195 for 100 consecutive episodes

## 🔄 How It Works

1. **Seeding**: Set random seeds for reproducibility
2. **Experience Collection**: Agent interacts with environment using ε-greedy policy
3. **Experience Replay**: Transitions are stored in a replay buffer
4. **Double DQN Training**: Random batches sampled and used to update Q-network
5. **Target Network**: Separate network provides stable targets, synced periodically
6. **Evaluation**: Greedy policy evaluated every 20 episodes
7. **Early Stopping**: Training stops when environment is solved

## 📚 References

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
- van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
