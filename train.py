import os
import traci
import numpy as np
import torch
import torch.optim as optim
from neural_net import PolicyNetwork
import matplotlib.pyplot as plt

SUMO_BINARY = "sumo"  # hoặc "sumo-gui" nếu muốn hiển thị
CONFIG_FILE = "intersection.sumocfg"
LANES = ["north_in_0", "east_in_0", "south_in_0", "west_in_0"]
PHASES = {"NSG": 0, "EWG": 1}
PHASE_DURATION = 5  # 5 giây theo paper
MAX_VEHICLES = 10  # ngưỡng tối đa xe mỗi làn
TERMINATION_THRESHOLD = 11  # ngưỡng chấm dứt nếu vượt quá

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state():
    state = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
    return torch.tensor([x / MAX_VEHICLES for x in state], dtype=torch.float32).to(device)

def compute_reward(state):
    std = torch.std(state).item()
    if std < 0.1:
        r = 1 - (std / 0.1)
    elif std >= 0.1:
        r = -1 * (std - 0.1) / 0.4
    return max(min(r, 1), -1)

def set_phase(action):
    if action == 0:
        traci.trafficlight.setPhase("center", PHASES["NSG"])
    else:
        traci.trafficlight.setPhase("center", PHASES["EWG"])

def run_episode(policy_net, optimizer, gamma=0.85, episode_num=0):
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
    log_probs = []
    rewards = []
    current_action = None
    action_step = 0
    phase_count = 0
    epsilon = max(0.05, 0.3 - 0.001 * episode_num)  # Decay from 0.3 to 0.05

    for step in range(250 * PHASE_DURATION):
        state = get_state()
        if step % PHASE_DURATION == 0 and step // PHASE_DURATION < 250:
            probs = policy_net(state)
            dist = torch.distributions.Categorical(probs)
            if np.random.random() < epsilon:
                current_action = torch.randint(0, 2, (1,)).item()
                log_probs.append(torch.tensor(0.0))
            else:
                current_action = dist.sample().item()
                log_probs.append(dist.log_prob(torch.tensor(current_action).to(device)))
            set_phase(current_action)
            action_step = step
            phase_count += 1

        traci.simulationStep()
        if step >= action_step + PHASE_DURATION - 1 and current_action is not None:
            next_state = get_state()
            reward = compute_reward(next_state)
            rewards.append(reward)
            current_action = None

        if step % 50 == 0:  # Log every 50 steps
            lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
            print(f"Step {step}, Lane vehicles: {lane_counts}")

        if max(traci.lane.getLastStepVehicleNumber(lane) for lane in LANES) > TERMINATION_THRESHOLD:
            break

    traci.close()
    if len(log_probs) > len(rewards):
        log_probs = log_probs[:len(rewards)]
    return log_probs, rewards

def train():
    policy_net = PolicyNetwork(input_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    num_episodes = 800
    update_freq = 10
    all_rewards = []

    for episode in range(num_episodes):
        log_probs, rewards = run_episode(policy_net, optimizer, gamma=0.9, episode_num=episode)
        all_rewards.append(np.sum(rewards) if rewards else 0)

        if log_probs and rewards and len(log_probs) == len(rewards):
            discounted_rewards = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.9 * G
                discounted_rewards.insert(0, G)

            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
            loss = -torch.stack(log_probs) * discounted_rewards
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards) if rewards else 0:.2f}")

    torch.save(policy_net.state_dict(), "model.pth")
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward over Time")
    plt.savefig("reward_plot.png")
    plt.show()

if __name__ == "__main__":
    train()