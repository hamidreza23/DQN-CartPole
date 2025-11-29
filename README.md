# DQN Implementation for CartPole

A complete Deep Q-Network (DQN) implementation for solving the CartPole-v1 environment using PyTorch. This implementation follows the original DQN paper by Mnih et al. (2015) with experience replay and target networks.

## 🚀 Features

- **DQN Network**: 3-layer fully connected network (128-128 neurons)
- **Experience Replay**: Replay buffer for stable training and data efficiency
- **Target Network**: Separate target network for stable Q-learning updates
- **ε-greedy Exploration**: Decaying epsilon for exploration-exploitation balance
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

## 🏗️ Architecture

### Environment
- **State Space**: 4 dimensions (CartPole observation: position, velocity, angle, angular velocity)
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

## 🔑 Key Components

1. **DQN Class**: Neural network for Q-value approximation
2. **ReplayBuffer**: Stores and samples experiences (state, action, reward, next_state, done)
3. **select_action**: ε-greedy action selection (exploration vs exploitation)
4. **train_step**: Performs one Bellman update step with target network
5. **train_dqn**: Complete training loop with statistics tracking

## 📊 Expected Performance

The agent should achieve an average reward of **200+** (CartPole's maximum) within **200-300 episodes**. The training output shows:
- Episode rewards
- Average reward over last 100 episodes
- Current epsilon (exploration rate)
- Training loss

## 🧠 How It Works

1. **Experience Collection**: Agent interacts with environment using ε-greedy policy
2. **Experience Replay**: Transitions are stored in a replay buffer
3. **Training**: Random batches are sampled and used to update the Q-network
4. **Target Network**: A separate network provides stable targets for Q-learning
5. **Periodic Updates**: Target network is synced with online network periodically

## 📚 References

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

