# TidyMesh Multi-Agent Simulation

A comprehensive multi-agent system for waste collection simulation featuring Q-Learning autonomous trucks, contract net protocol negotiation, and dynamic visualization.

## Project Structure

```
TidyMesh_Sim/
├── TidyMesh_Sim.py          # Main simulation file
├── visualizer.py            # Comprehensive visualization system
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── results/                 # Simulation output files
│   ├── simulation_data/     # JSON data files
│   │   ├── mas_final_state.json      # Final simulation state
│   │   └── simulation_history.json   # Step-by-step history
│   └── visualizations/      # Generated plots and animations
│       ├── simulation_overview.png   # Static overview charts
│       ├── qlearning_analysis.png    # Q-Learning analysis
│       └── simulation_animation.gif  # Animated simulation
│
├── documentation/           # Project documentation
│   └── TidyMesh_QLearning_Analysis_Report.html  # Q-Learning analysis report
│
└── scripts/                # Utility scripts
    ├── debug_history.py     # Animation debugging tool
    ├── generate_qlearning_pdf.py  # PDF report generator (legacy)
    └── simple_qlearning_report.py # HTML report generator
```

## Features

### Core Simulation
- **Multi-Agent System**: 5 garbage trucks, 20 waste bins, traffic lights, dynamic obstacles
- **Q-Learning**: Reinforcement learning for autonomous truck navigation
- **Contract Net Protocol**: Distributed task allocation between dispatcher and trucks
- **Dynamic Environment**: Moving obstacles, traffic light cycles, evolving bin states

### Visualization System
- **Static Overview**: Grid layout, truck statistics, bin status, performance metrics
- **Q-Learning Analysis**: Action distribution, learning progress, efficiency metrics, spatial analysis  
- **Animated Simulation**: Real-time GIF animation of the complete simulation

### Documentation
- **Comprehensive Analysis**: Detailed Q-Learning implementation report
- **Technical Documentation**: Cliff conditions, learning patterns, hybrid approach analysis

## Quick Start

### 1. Run the Simulation
```bash
python TidyMesh_Sim.py
```

### 2. Generate Visualizations
```bash
python visualizer.py
```

### 3. Create Documentation
```bash
cd scripts
python simple_qlearning_report.py
```

## Key Parameters

- **Simulation Duration**: 3600 steps (6-minute simulation)
- **Grid Size**: 20x14 cells
- **Agents**: 5 trucks, 20 bins, 10 traffic lights, 15 obstacles
- **Q-Learning**: α=0.5, γ=0.90, ε=0.05

## File Outputs

### Simulation Data
- `mas_final_state.json`: Final positions and states of all agents
- `simulation_history.json`: Complete step-by-step simulation history

### Visualizations
- `simulation_overview.png`: Multi-panel static analysis
- `qlearning_analysis.png`: Q-Learning behavior analysis  
- `simulation_animation.gif`: Animated simulation playback

### Documentation
- `TidyMesh_QLearning_Analysis_Report.html`: Comprehensive technical analysis

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- agentpy (multi-agent framework)
- matplotlib (plotting)
- seaborn (statistical visualization)
- pandas (data processing)
- pillow (image processing)
- numpy (numerical computation)

## Architecture Overview

### Agent Types
- **GarbageTruck**: Q-Learning enabled autonomous agents
- **TrashBin**: Dynamic task sources with fill level simulation
- **Dispatcher**: Contract Net coordinator using fairness algorithms
- **TrafficLight**: Periodic state cycling (red/green)
- **DynamicObstacle**: Random movement patterns
- **Depot**: Waste unloading location

### Learning System
- **Hybrid Approach**: Combines Q-Learning exploration with deterministic pathfinding
- **State Space**: Grid positions with contextual information
- **Action Space**: Movement directions plus operational actions
- **Reward Structure**: Task completion, efficiency, and collision avoidance

### Visualization Engine
- **Non-Interactive Backend**: Server-compatible matplotlib configuration
- **Multi-Format Output**: PNG static plots, GIF animations
- **Progress Tracking**: Real-time feedback during generation
- **Error Handling**: Robust failure recovery and reporting

## Usage Examples

### Run with Custom Parameters
```python
from TidyMesh_Sim import CityWasteModel

# Custom configuration
params = {
    "n_trucks": 3,
    "n_bins": 15,
    "steps": 1800,
    "q_alpha": 0.3,
    "q_epsilon": 0.1
}

model = CityWasteModel(params)
results = model.run()
```

### Generate Specific Visualizations
```python
from visualizer import TidyMeshVisualizer

viz = TidyMeshVisualizer()
viz.create_static_overview()        # Static charts only
viz.create_qlearning_analysis()     # Q-Learning analysis only
viz.create_animated_simulation()    # Animation only
```

## Development Notes

- **Performance**: Simulation runs in real-time with 0.1s step delays
- **Scalability**: Current implementation supports up to 50 agents efficiently
- **Extensibility**: Modular design allows easy addition of new agent types
- **Debugging**: Comprehensive logging and visualization tools included

## License

Academic project for Multi-Agent Computer Graphics course.

## Authors

Santiago & Development Team  
Universidad - 5th Semester  
Multi-Agent Computer Graphics Course
