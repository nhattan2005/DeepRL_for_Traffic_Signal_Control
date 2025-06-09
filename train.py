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
PHASE_DURATION = 5  # 5 giây mỗi pha
MAX_STEPS = 750  # Số bước tối đa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state():
    # Trả về trạng thái đơn giản: số phương tiện trên mỗi làn
    state = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
    return torch.tensor(state, dtype=torch.float32).to(device)

def compute_reward(state, previous_action=None, current_action=None, phase_duration=0):
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

def run_episode(policy_net, optimizer, gamma=0.9, episode_num=0):
    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
        log_probs = []
        rewards = []
        current_action = None
        action_step = 0
        epsilon = max(0.4, 0.95 - 0.0005 * episode_num)  # Giảm từ 0.95 xuống 0.4 rất chậm

        for step in range(MAX_STEPS):
            state = get_state()
            if step % PHASE_DURATION == 0:
                probs = policy_net(state)
                dist = torch.distributions.Categorical(probs)
                if np.random.random() < epsilon:
                    current_action = torch.randint(0, 2, (1,)).item()
                else:
                    current_action = dist.sample().item()
                log_probs.append(dist.log_prob(torch.tensor(current_action, device=device)))
                set_phase(current_action)
                action_step = step

            traci.simulationStep()
            if step >= action_step + PHASE_DURATION - 1 and current_action is not None:
                next_state = get_state()
                reward = compute_reward(next_state)
                rewards.append(reward)
                current_action = None

            # Xử lý teleport
            if traci.simulation.getEndingTeleportNumber() > 0:
                penalty = -5.0
                rewards.append(penalty)
                current_action = None

            if step % 50 == 0:
                lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
                print(f"Step {step}, Lane vehicles: {lane_counts}")

        traci.close()
    except Exception as e:
        print(f"Error during simulation: {e}")
        traci.close()
        return [], []

    if len(log_probs) > len(rewards):
        log_probs = log_probs[:len(rewards)]  # Cắt log_probs để khớp với rewards
    return log_probs, rewards

def train():
    policy_net = PolicyNetwork(input_dim=4, output_dim=2).to(device)  # 4 lanes, 2 actions
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    num_episodes = 500
    all_rewards = []

    for episode in range(num_episodes):
        log_probs, rewards = run_episode(policy_net, optimizer, episode_num=episode)
        all_rewards.append(sum(rewards) if rewards else 0)

        if log_probs and rewards and len(log_probs) == len(rewards):
            discounted_rewards = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.9 * G
                discounted_rewards.insert(0, G)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
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