import os
import traci
import numpy as np
import time
import torch
import torch.optim as optim
from neural_net import PolicyNetwork
import matplotlib.pyplot as plt

# ====== Cấu hình SUMO và môi trường ======
SUMO_BINARY = "sumo"  # hoặc "sumo-gui" nếu muốn hiển thị đồ họa
CONFIG_FILE = "intersection.sumocfg"
LANES = ["north_in_0", "east_in_0", "south_in_0", "west_in_0"]
PHASES = {"NSG": 0, "EWG": 2}
PHASE_DURATION = 5  # Thời gian giữ pha, 5 giây như paper
MAX_VEHICLES = 5    # Ngưỡng tối đa xe mỗi làn
MAX_STEPS = 750      # Số bước mô phỏng mỗi episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Hàm lấy trạng thái môi trường ======
def get_state():
    state = [min(traci.lane.getLastStepVehicleNumber(lane) / MAX_VEHICLES, 1.0) for lane in LANES]
    return torch.tensor(state, dtype=torch.float32).to(device)

# ====== Hàm tính phần thưởng theo paper ======
def compute_reward(state):
    std = torch.std(state).item()
    if std < 0.1:
        reward = 1.0 - (std / 0.1)
    else:
        reward = -((std - 0.1) / (0.5 - 0.1))
    return max(min(reward, 1.0), -1.0)

# ====== Hàm điều khiển pha đèn giao thông ======
def set_phase(action):
    traci.trafficlight.setPhase("center", PHASES["NSG"] if action == 0 else PHASES["EWG"])

# ====== Chạy một episode ======
def run_episode(policy_net, episode_num=0):
    log_probs = []
    rewards = []
    states = []
    current_action = None
    action_step = 0
    previous_action = None

    # ✅ Reset simulation mà không cần restart GUI
    traci.load(["-c", CONFIG_FILE])

    for step in range(MAX_STEPS):
        state = get_state()
        if step % PHASE_DURATION == 0:
            probs = policy_net(state)
            dist = torch.distributions.Categorical(probs)

            # Epsilon-greedy exploration (nếu muốn thêm)
            epsilon = max(0.05, 0.2 - 0.0001 * episode_num)
            if np.random.rand() < epsilon:
                current_action = np.random.choice([0, 1])
            else:
                current_action = dist.sample().item()

            log_probs.append(dist.log_prob(torch.tensor(current_action, device=device)))
            states.append(state)
            set_phase(current_action)
            action_step = step
            previous_action = current_action

            # print(f"Step {step}, Phase: {'NSG' if current_action == 0 else 'EWG'}, STD: {torch.std(state).item():.4f}, Reward: {compute_reward(state):.4f}")

        traci.simulationStep()
        # time.sleep(0.05)  # ✅ Cho GUI dễ nhìn hơn

        if step >= action_step + PHASE_DURATION - 1:
            next_state = get_state()
            reward = compute_reward(next_state)
            rewards.append(reward)

        lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
        if max(lane_counts) > MAX_VEHICLES:
            rewards.append(-20.0)
            print(f"Terminating episode early at step {step} due to overload. Lane counts: {lane_counts}")
            break

        if current_action is not None and (step - action_step) >= PHASE_DURATION:
            if any(count > 0 for count in lane_counts) and any(count == 0 for count in lane_counts):
                if (current_action == 0 and max(lane_counts[1], lane_counts[3]) > 0 and lane_counts[0] == 0 and lane_counts[2] == 0) or \
                (current_action == 1 and max(lane_counts[0], lane_counts[2]) > 0 and lane_counts[1] == 0 and lane_counts[3] == 0):
                    rewards.append(-5.0)
                    print(f"Terminating episode early at step {step} due to wrong phase choice. Action: {current_action}, Lane counts: {lane_counts}")
                    break

        if step % 50 == 0:
            print(f"Step {step}, Lane vehicles: {lane_counts}")

    if len(log_probs) > len(rewards):
        log_probs = log_probs[:len(rewards)]
        states = states[:len(rewards)]

    return log_probs, rewards, states


# ====== Huấn luyện mô hình ======
def train():
    policy_net = PolicyNetwork(input_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    num_episodes = 500
    all_rewards = []
    update_freq = 10

    # ✅ Chỉ mở GUI một lần duy nhất
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    for episode in range(num_episodes):
        log_probs, rewards, states = run_episode(policy_net, episode_num=episode)
        total_reward = sum(rewards) if rewards else 0
        all_rewards.append(np.mean(rewards) if rewards else 0)

        if log_probs and rewards and len(log_probs) == len(rewards):
            discounted_rewards = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.9 * G
                discounted_rewards.insert(0, G)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)

            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

            entropies = []
            for state in states:
                probs = policy_net(state)
                dist = torch.distributions.Categorical(probs)
                entropies.append(dist.entropy())
            entropy = torch.stack(entropies).mean()

            loss = (-torch.stack(log_probs) * discounted_rewards).sum() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % update_freq == 0:
            print(f"Episode {episode}, Total: {total_reward:.2f}, Avg/Step: {np.mean(rewards):.2f}")

    traci.close()  # ✅ Đóng GUI sau khi train xong

    torch.save(policy_net.state_dict(), "model.pth")
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label="Average Reward per Step")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Training Progress")
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_plot.png")
    plt.show()


if __name__ == "__main__":
    train()
