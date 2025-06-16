# ------------------------------
# Imports
# ------------------------------

import torch
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ------------------------------
# Detect3DeepRL
# ------------------------------

class Detect3DeepRL:
    def __init__(self,
                 max_steps=100,
                 memory_size=10000,
                 batch_size=32,
                 learning_rate=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.99):

        self.max_steps = max_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ---------------------------
        # 2. Markov Decision Process
        # ---------------------------
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.state_space = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        self.action_space = {0: "NO", 1: "YES"}

        # --------------------------
        # 4. Simulation Environment
        # --------------------------
        self.current_step = 0
        self.current_index = None
        self.done = False

        # --------------------------
        # 5. Defining the Model
        # --------------------------
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self._sync_target_network()

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

        self.memory = deque(maxlen=memory_size)

        # --------------------------
        # Metric Visualization Setup
        # --------------------------
        self.metric_steps = []
        self.metric_accuracy = []
        self.metric_precision = []
        self.metric_recall = []
        self.metric_f1 = []


    def reset(self):
        self.current_step = 0
        self.done = False
        self.current_index = random.randint(0, len(self.state_space) - 1)
        image, label = self.state_space[self.current_index]
        label = 1 if label == 3 else 0
        return image.unsqueeze(0), label

    def step(self, action):
        if self.done:
            raise Exception("Episode is done. Call reset().")

        _, true_label = self.state_space[self.current_index]
        true_label = 1 if true_label == 3 else 0

        reward = 1 if action == true_label else -1

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        self.current_index = random.randint(0, len(self.state_space) - 1)
        next_image, next_label = self.state_space[self.current_index]
        next_label = 1 if next_label == 3 else 0

        return next_image.unsqueeze(0), reward, self.done, {"label": next_label}

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(36864, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def _sync_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # --------------------------
    # 7. Training the Model
    # --------------------------
    def train(self, num_episodes=10):
        for episode in range(1, num_episodes + 1):
            if episode % 10 == 0 or episode == 1:
                print(f"Training episode {episode}/{num_episodes}...")

            state, _ = self.reset()
            done = False

            while not done:
                if random.random() < self.epsilon:
                    action = random.choice(list(self.action_space.keys()))
                else:
                    with torch.no_grad():
                        q_values = self.q_network(state)
                        action = torch.argmax(q_values).item()

                next_state, reward, done, _ = self.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(self.memory) >= self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.cat(states)
                    next_states = torch.cat(next_states)
                    actions = torch.tensor(actions)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    dones = torch.tensor(dones, dtype=torch.bool)

                    q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(1)[0]
                        targets = rewards + (1 - dones.float()) * self.gamma * next_q_values

                    loss = self.criterion(q_values, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self._sync_target_network()

    # --------------------------
    # 8. Testing the Model
    # --------------------------
    def test(self, num_samples=20):
        test_data = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
        true_labels = []
        predicted_labels = []

        for i in range(num_samples):
            image, label = test_data[i]
            label_bin = 1 if label == 3 else 0
            image = image.unsqueeze(0)

            with torch.no_grad():
                q_vals = self.q_network(image)
                action = torch.argmax(q_vals).item()

            true_labels.append(label_bin)
            predicted_labels.append(action)

            acc = accuracy_score(true_labels, predicted_labels)
            prec = precision_score(true_labels, predicted_labels, zero_division=0)
            rec = recall_score(true_labels, predicted_labels, zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, zero_division=0)

            self.metric_steps.append(i + 1)
            self.metric_accuracy.append(acc)
            self.metric_precision.append(prec)
            self.metric_recall.append(rec)
            self.metric_f1.append(f1)

            clear_output(wait=True)
            plt.figure(figsize=(14, 8))

            plt.subplot(2, 2, 1)
            plt.plot(self.metric_steps, self.metric_accuracy)
            plt.title("Accuracy")
            plt.ylim(0, 1)
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(self.metric_steps, self.metric_precision)
            plt.title("Precision")
            plt.ylim(0, 1)
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(self.metric_steps, self.metric_recall)
            plt.title("Recall")
            plt.ylim(0, 1)
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(self.metric_steps, self.metric_f1)
            plt.title("F1 Score")
            plt.ylim(0, 1)
            plt.grid(True)

            plt.tight_layout()
            plt.show()


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    # ✅ PARAMETERS YOU CAN TUNE:
    config = {
        # Maximum number of steps per episode during training.
        # The agent will reset after this number of steps.
        # Typical values: 10 to 100 depending on task complexity.
        'max_steps': 10,

        # Maximum number of experiences stored in the replay memory.
        # Once the limit is reached, old experiences are discarded.
        # Typical values: 1,000 to 100,000.
        'memory_size': 5000,

        # Number of experiences sampled from memory for each training update.
        # Affects training stability and GPU/CPU load.
        # Common values: 16, 32, 64, 128.
        'batch_size': 32,

        # Learning rate for the optimizer.
        # Controls how fast the model weights are updated.
        # Typical values: 1e-4 to 1e-2. Lower for stable learning.
        'learning_rate': 1e-3,

        # Discount factor for future rewards.
        # 0.0 means agent only cares about immediate reward.
        # 0.99 means agent highly values long-term rewards.
        # Values range between 0.0 and 1.0.
        'gamma': 0.0,

        # Initial value for epsilon in epsilon-greedy policy.
        # Controls how much the agent explores randomly at the beginning.
        # Typical start: 1.0 (100% random actions).
        'epsilon_start': 1.0,

        # Minimum value that epsilon can decay to.
        # Ensures the agent always keeps some exploration.
        # Common values: 0.01 to 0.1.
        'epsilon_min': 0.1,

        # Decay rate for epsilon after each episode.
        # Controls how fast the agent shifts from exploration to exploitation.
        # Typical values: 0.90 to 0.995.
        'epsilon_decay': 0.99,

        # Number of training episodes the agent will go through.
        # More episodes usually result in better learning.
        # Typical values: 100 to 10,000 depending on problem complexity.
        # Set to 500 for a good middle ground.  But 100 will be quicker.
        'train_episodes': 500,

        # Number of test samples used during evaluation.
        # Affects the resolution of performance metrics.
        # Common range: 100 to 10,000.
        'test_samples': 200
    }


    agent = Detect3DeepRL(
        max_steps=config['max_steps'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay']
    )

    total_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)

    print("✅ Agent initialization complete.")
    print(f"Training dataset size: {len(agent.state_space)} images.")

    print("\n🚀 Running training phase... Please wait.")
    agent.train(num_episodes=config['train_episodes'])

    print("\n📊 Training complete. Running test phase and generating metric plots...")
    agent.test(num_samples=config['test_samples'])

    print("\n✅ Agent execution finished.")
    print(f"\nTotal number of learnable parameters (neurons): {total_params}")

    # Save trained Q-network
    checkpoint = {
        "model_state_dict": agent.q_network.state_dict(),
        "config": config,
        "epsilon": agent.epsilon
    }

    torch.save(checkpoint, "detect3_training_checkpoint.pth")
    print("✅ Full checkpoint saved as detect3_training_checkpoint.pth")