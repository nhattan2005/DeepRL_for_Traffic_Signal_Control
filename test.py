import traci
import torch
import numpy as np
from neural_net import PolicyNetwork

SUMO_BINARY = "sumo-gui"  # dùng GUI để quan sát
CONFIG_FILE = "intersection.sumocfg"
LANES = ["north_in_0", "east_in_0", "south_in_0", "west_in_0"]
TRAFFIC_LIGHT_ID = "center"
PHASE_DURATION = 5  # 5 giây theo paper
MAX_VEHICLES = 10  # chuẩn hóa
MAX_STEPS = 2000  # giới hạn số bước để tránh quá tải

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state():
    state = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
    return torch.tensor([x / MAX_VEHICLES for x in state], dtype=torch.float32).to(device)

def set_phase(action):
    phase_mapping = {0: 0, 1: 2}  # NS Green, EW Green
    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, phase_mapping[action])

def test():
    policy_net = PolicyNetwork(input_dim=4, output_dim=2).to(device)
    policy_net.load_state_dict(torch.load("model.pth"))
    policy_net.eval()

    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    for step in range(MAX_STEPS):
        state = get_state()
        with torch.no_grad():
            probs = policy_net(state)
            if np.random.random() < 0.3:  # Tăng lên 30% cơ hội chọn ngẫu nhiên
                action = torch.randint(0, 2, (1,)).item()
            else:
                action = torch.argmax(probs).item()

        if step % PHASE_DURATION == 0:  # chỉ cập nhật pha mỗi 5 giây
            set_phase(action)
            print(f"Step {step}, Chosen Phase: {'NSG' if action == 0 else 'EWG'}")

        traci.simulationStep()

        # Theo dõi số xe mỗi 50 bước
        if step % 50 == 0:
            lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
            print(f"Step {step}, Lane vehicles: {lane_counts}")

        # Kiểm tra kết thúc sớm
        lane_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in LANES]
        if max(lane_counts) > MAX_VEHICLES:
            print(f"Simulation ended at Step {step} due to max vehicles exceeded.")
            break

    traci.close()
    print("Simulation completed.")

if __name__ == "__main__":
    test()