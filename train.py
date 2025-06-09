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
PHASE_DURATION = 5  # 5 giây mỗi pha, theo paper
MAX_VEHICLES = 5  # Giảm ngưỡng tối đa số xe mỗi làn
MAX_STEPS = 750  # Số bước tối đa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state():
    state = [min(traci.lane.getLastStepVehicleNumber(lane) / MAX_VEHICLES, 1.0) for lane in LANES]
    return torch.tensor(state, dtype=torch.float32).to(device)

def compute_reward(state, previous_action=None, current_action=None, step=0, action_step=0):
    std = torch.std(state).item()
    if std < 0.1:
        r_max, r_min, sigma_max, sigma_min = 1.0, 0.0, 0.1, 0.0
    else:
        r_max, r_min, sigma_max, sigma_min = 0.0, -1.0, 0.5, 0.1
    reward = r_max + ((std - sigma_min) * (r_min - r_max)) / (sigma_max - sigma_min)
    
    # Phạt mạnh nếu giữ pha quá 10 giây (2 pha)
    if previous_action is not None and current_action == previous_action and (step - action_step) // PHASE_DURATION >= 2:
        reward -= 10.0
    # Thêm phần thưởng khi chuyển pha
    elif previous_action is not None and current_action != previous_action:
        reward += 1.0
    return max(min(reward, 1), -1)

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
        previous_action = None
        epsilon = max(0.5, 0.95 - 0.00002 * episode_num)  # Duy trì 0.5 làm mức tối thiểu

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
                previous_action = current_action if previous_action is None else current_action
                print(f"Step {step}, Chosen Phase: {'NSG' if current_action == 0 else 'EWG'}")

            traci.simulationStep()
            if step >= action_step + PHASE_DURATION - 1 and current_action is not None:
                next_state = get_state()
                reward = compute_reward(next_state, previous_action, current_action, step, action_step)
                rewards.append(reward)
                current_action = None
                previous_action = None

            # Kiểm tra điều kiện kết thúc
            lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
            if max(lane_counts) > MAX_VEHICLES:
                rewards.append(-10.0)
                break
            if current_action is not None:
                if any(count > 0 for count in lane_counts) and any(count == 0 for count in lane_counts):
                    if (current_action == 0 and max(lane_counts[1], lane_counts[3]) > 0 and lane_counts[0] == 0 and lane_counts[2] == 0) or \
                       (current_action == 1 and max(lane_counts[0], lane_counts[2]) > 0 and lane_counts[1] == 0 and lane_counts[3] == 0):
                        rewards.append(-10.0)
                        break

            if step % 50 == 0:
                lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
                print(f"Step {step}, Lane vehicles: {lane_counts}")

        traci.close()
    except Exception as e:
        print(f"Error during simulation: {e}")
        traci.close()
        return [], []

    if len(log_probs) > len(rewards):
        log_probs = log_probs[:len(rewards)]
    return log_probs, rewards

def train():
    policy_net = PolicyNetwork(input_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    num_episodes = 500
    all_rewards = []
    update_freq = 10

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

        if episode % update_freq == 0 and episode > 0:
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