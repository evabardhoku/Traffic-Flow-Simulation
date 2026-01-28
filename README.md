# Traffic Flow Simulation & Control (Macroscopic Modeling)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-orange?style=for-the-badge&logo=mathworks)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Project Overview
This repository contains a comparative simulation of highway traffic flow using **Macroscopic Modeling (METANET)**. The project analyzes traffic density evolution and implements control strategies like **ALINEA (Ramp Metering)** and **Semaphores** to mitigate congestion in bottleneck scenarios.

The simulation compares two implementation approaches:
1.  **Python Implementation:** Custom simulation loop using `NumPy` and `Matplotlib`.
2.  **MATLAB Implementation:** Validated simulation scripts for rapid prototyping.

---

## Theoretical Background
The simulation is grounded in the **METANET** model, a second-order macroscopic traffic flow model that describes the spatiotemporal evolution of:
- **Density ($\rho$):** Vehicles per kilometer.
- **Speed ($v$):** Mean speed of the traffic stream.
- **Flow ($q$):** Traffic volume ($q = \rho \cdot v$).

### Control Strategy: ALINEA
To optimize flow at on-ramps, we utilized the **ALINEA** (Asservissement Linéaire d'Entrée Autoroutière) algorithm, a local feedback control law:

$$r(k) = r(k-1) + K_R [\hat{o} - o_{out}(k)]$$

Where:
- $r(k)$ is the metering rate.
- $\hat{o}$ is the target occupancy (critical density).
- $K_R$ is the regulator parameter.

---

## Scenarios Analysis

We simulated three distinct traffic scenarios to evaluate the efficiency of the control algorithms:

### 1. No Control (Baseline)
- **Scenario:** High traffic demand enters from the on-ramp without restriction.
- **Observation:** Traffic density spikes at the merge point, causing a shockwave that propagates backward (congestion).
- **Result:** High total travel time and reduced network capacity.

### 2. Semaphore Control (Fixed Cycle)
- **Scenario:** A simple traffic light at the on-ramp with fixed Red/Green phases.
- **Observation:** Reduces the inflow rate but fails to adapt to dynamic changes in mainline traffic.
- **Result:** Slight improvement over baseline, but suboptimal during peak variations.

### 3. ALINEA Control (Ramp Metering) 
- **Scenario:** Dynamic adjustment of the on-ramp flow based on real-time feedback from downstream sensors.
- **Observation:** The controller successfully restricts ramp flow when mainline density approaches critical levels ($\rho_{cr}$).
- **Result:** Maintained fluid flow, prevented traffic breakdown, and maximized total throughput.

---

## Getting Started

### Prerequisites
- **Python 3.x** (with `numpy`, `matplotlib`, `pandas`)
- **MATLAB** (R2020a or later recommended)

---

## Author: Eva Bardhoku

### Running the Python Simulation
```bash
# 1. Clone the repository
git clone [https://github.com/evabardhoku/Traffic-Flow-Simulation.git](https://github.com/evabardhoku/Traffic-Flow-Simulation.git)

# 2. Navigate to the Python directory
cd Traffic-Flow-Simulation/Python_Code

# 3. Run the main simulation
python simulation.py



