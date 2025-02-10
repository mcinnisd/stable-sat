import gym
from gym import spaces
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# 1. Define the Custom Environment
# ==============================
class SatelliteAttitudeEnv(gym.Env):
    """
    A simple simulation for one-axis satellite attitude control.
    State:
        - theta: Angular position (radians)
        - theta_dot: Angular velocity (radians/second)
    Action:
        - Discrete torque commands: 0 -> -1, 1 -> 0, 2 -> +1 (arbitrary units)
    Dynamics:
        theta_dot_next = theta_dot + (torque / inertia - damping * theta_dot) * dt
        theta_next = theta + theta_dot_next * dt
    Reward:
        Negative quadratic penalty for deviation from zero (both angle and angular velocity).
    """
    def __init__(self):
        super(SatelliteAttitudeEnv, self).__init__()
        # Define action space: 3 discrete actions corresponding to torque values
        self.action_space = spaces.Discrete(3)
        # Observation space: [theta, theta_dot] with reasonable bounds
        high = np.array([np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Physical parameters
        self.dt = 0.1               # time step (seconds)
        self.inertia = 1.0          # moment of inertia
        self.damping = 0.1          # damping coefficient
        self.torque_vals = [-1.0, 0.0, 1.0]  # mapping from discrete action to torque
        
        self.max_steps = 200        # maximum steps per episode
        self.current_step = 0
        
        self.state = None

    def reset(self):
        """
        Reset the environment to an initial state.
        Here we randomly initialize the angular position and velocity.
        """
        theta = np.random.uniform(-np.pi/4, np.pi/4)
        theta_dot = np.random.uniform(-1.0, 1.0)
        self.state = np.array([theta, theta_dot], dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        """
        Apply the action and update the state.
        """
        torque = self.torque_vals[action]
        theta, theta_dot = self.state

        # Calculate angular acceleration
        theta_acc = (torque / self.inertia) - (self.damping * theta_dot)
        
        # Update state using Euler integration
        theta_dot_new = theta_dot + theta_acc * self.dt
        theta_new = theta + theta_dot_new * self.dt

        # Clip values to observation space limits
        theta_new = np.clip(theta_new, self.observation_space.low[0], self.observation_space.high[0])
        theta_dot_new = np.clip(theta_dot_new, self.observation_space.low[1], self.observation_space.high[1])
        self.state = np.array([theta_new, theta_dot_new], dtype=np.float32)
        
        # Calculate reward (penalize the squared error from zero)
        reward = - (theta_new**2 + 0.1 * theta_dot_new**2)
        
        # Increment step count and determine if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Optionally, add a condition to end the episode if the satellite is well stabilized:
        if np.abs(theta_new) < 0.01 and np.abs(theta_dot_new) < 0.01:
            done = True
            reward += 10.0  # bonus for stabilization
        
        return self.state, reward, done, {}

    def render(self, mode='human'):
        """
        For simplicity, we print the state.
        """
        print(f"Step: {self.current_step}, State: {self.state}")

# ==============================
# 2. Define the Replay Memory
# ==============================
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# ==============================
# 3. Define the Q-Network using PyTorch
# ==============================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ==============================
# 4. Define Utility Functions for the DQN Agent
# ==============================
def select_action(state, policy_net, epsilon, device):
    """
    Select an action using an epsilon-greedy policy.
    """
    if random.random() < epsilon:
        # Explore: select a random action
        return random.randrange(env.action_space.n)
    else:
        # Exploit: select the action with max Q-value
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    """
    Sample a batch from memory and perform a gradient descent step.
    """
    if len(memory) < batch_size:
        return
    
    # Sample a random batch of transitions
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    
    # Convert batches to tensors
    batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
    batch_action = torch.tensor(batch_action, dtype=torch.int64, device=device).unsqueeze(1)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
    batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
    batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Compute current Q-values
    current_q_values = policy_net(batch_state).gather(1, batch_action)
    
    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        # If done, there is no next Q value.
        target_q_values = batch_reward + gamma * max_next_q_values * (1 - batch_done)
    
    # Compute loss (mean squared error)
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==============================
# 5. Training Loop for the DQN Agent
# ==============================
if __name__ == '__main__':
    # Create the environment
    env = SatelliteAttitudeEnv()
    
    # Check device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    num_episodes = 500            # total training episodes
    batch_size = 64               # minibatch size for DQN
    gamma = 0.99                  # discount factor
    epsilon_start = 1.0           # initial exploration rate
    epsilon_end = 0.05            # final exploration rate
    epsilon_decay = 300           # decay rate for epsilon
    target_update_freq = 10       # update target network every N episodes
    learning_rate = 1e-3
    memory_capacity = 10000       # replay memory capacity

    # Initialize policy and target networks
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # set target network to evaluation mode
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    
    # Main training loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        
        for t in range(env.max_steps):
            # Epsilon decay schedule
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
            
            # Select and perform an action
            action = select_action(state, policy_net, epsilon, device)
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            
            # Store transition in replay memory
            memory.push(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            
            # Perform optimization step
            optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma, device)
            
            if done:
                break
        
        # Update the target network periodically
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
    
    print("Training complete!")

    # ==============================
    # 6. Testing the Trained Policy
    # ==============================
    state = env.reset()
    done = False
    print("\nTesting the trained policy:")
    while not done:
        env.render()  # Print the current state
        action = select_action(state, policy_net, epsilon=0.0, device=device)  # No exploration during testing
        state, reward, done, _ = env.step(action)
