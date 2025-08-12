# UbABC: Utilization-Based Artificial Bee Colony for VM Placement

**Efficient VM Placement via Utilization-Based ABC**

---

##  Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithm Overview](#algorithm-overview)
- [Configuration](#configuration)
- [Results & Visualization](#results--visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
This repository houses the Python implementation of the **Utilization-Based Artificial Bee Colony (UbABC)** algorithm for efficient Virtual Machine (VM) placement on Physical Machines (PMs). The approach optimizes resource utilization, balances load, reduces active PM count, and minimizes migration cost.

---

## Features
- Core implementation of the UbABC metaheuristic for VM Placement  
- Baseline traditional ABC for comparative analysis  
- Modular code structure with support for random and file-based inputs  
- Automatic generation of convergence and utilization visualizations  
- Preconfigured parameter settings for easy experimentation

---

## Getting Started

### Prerequisites
Ensure your environment includes:
- **Python 3.8 or newer**
- Required packages: `numpy`, `matplotlib`

### Installation
```bash
git clone https://github.com/fatmaomar2/UbABC.git
cd UbABC
pip install -r requirements.txt
Usage
Run the baseline ABC:


python abc_algorithm.py --mode baseline
Run the UbABC optimized algorithm:


python abc_algorithm.py --mode ubabc
Visual comparisons are generated in the output/ directory, including convergence plots and resource utilization charts.

Project Structure

UbABC/
├── abc_algorithm.py        # Experiment runner & visualization
├── task3edit.py            # Core algorithm logic and data models
├── requirements.txt        # Python dependencies
└── output/                 # Generated plots and summaries
Algorithm Overview
The implementation follows the classic ABC structure with extensions:

Initialization: Solutions encoded via VM-to-PM assignment vectors.

Employed Bee Phase: Apply migration or swap operators for local improvement.

Onlooker Bee Phase: Probabilistic selection based on fitness; further refinement.

Scout Bee Phase: Replace stagnated solutions to maintain exploration.

Evaluation: Multi-objective fitness considering utilization, balance, energy, and migration.

Refer to the manuscript’s Section 5.3 for a detailed pseudo-code and algorithmic breakdown.

Configuration
Parameters are defined within task3edit.py:


num_employed_bees = 10
num_onlooker_bees = 10
max_iterations = 100
limit = 20
w1, w2, w3 = 0.4, 0.4, 0.2
Adjust these values to customize exploration depth or convergence behavior.

Results & Visualization
Upon execution, UbABC outputs:

Convergence plots (fitness over iterations)

Active PM usage charts

Resource utilization summaries

Comparative CSV (comparison_summary.csv) for statistical analysis

These visuals are instrumental for performance evaluation and presentation.

Contributing
Contributions are welcome!
Start by forking the project and submitting pull requests. For major changes, please open an issue to discuss your intended improvements.

