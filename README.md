# 🚦 Deep Reinforcement Learning for Traffic Signal Control

This project reproduces the method described in the paper:  
**"Deep Reinforcement Learning based approach for Traffic Signal Control"**  
Using a simple 4-way intersection in SUMO and PyTorch.

## 📁 Project Structure

```
.
├── intersection.sumocfg       # Main simulation config
├── intersection.net.xml       # Road network
├── intersection.rou.xml       # Route file (traffic flows)
├── intersection.tll.xml       # Traffic light logic (with 4 phases)
├── neural_net.py              # PyTorch PolicyNetwork
├── train.py                   # Training script
├── test.py                    # Testing script
├── test_case_1.rou.xml
├── test_case_2.rou.xml        
├── reward_plot.png            # Reward over time (after training)
└── README.md
```

## 🚀 Getting Started

### Train the Model (No GUI - Faster)

```bash
python train.py
```

If you want to see the GUI while training, update `train.py`:

```python
SUMO_BINARY = "sumo-gui"
```

Then run:

```bash
python train.py
```

## ⚙️ Traffic Light Phases

The simulation uses 4 phases:

- **NSG**: North–South Green
- **NSY**: North–South Yellow
- **EWG**: East–West Green
- **EWY**: East–West Yellow

Your agent controls **NSG** (phase 0) and **EWG** (phase 2) only. Yellow phases are managed automatically by SUMO.

## 🧲 How to Modify Test Cases

To use a different `.rou.xml` test case:

Update `intersection.sumocfg`:

```xml
<input>
    <net-file value="intersection.net.xml"/>
    <route-files value="test_case_1.rou.xml"/>
</input>
```

### Example: One-Direction Flow (East Only)

```xml
<flow id="east_west" type="car" from="east_in" to="west_out" begin="0" end="3600" period="10"/>
```

Create your own `.rou.xml` files to test edge cases.

## 📈 Result

After training, a plot will be saved:  
**reward_plot.png**

It shows the average reward per step over training episodes.

## 📌 Notes

- For best training performance, use `sumo` instead of `sumo-gui`.
- GUI is helpful for debugging but slows down training.
