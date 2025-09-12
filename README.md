# 🚀 TidyMesh Enhanced Multi-Agent Simulation

A comprehensive multi-agent system for urban waste collection simulation featuring **advanced Q-Learning**, **contract net protocol negotiation**, **coordinate transformation**, and **Unity integration** with RESTful API.

## 🔥 **KEY FEATURES v2.0**

### ✅ **Multi-Agent System Architecture**
- **Intelligent Garbage Trucks**: Q-Learning enabled agents with dynamic pathfinding
- **Contract Net Protocol**: Distributed task allocation with bidding system
- **Smart Dispatcher**: Centralized coordination with fallback assignment mechanisms  
- **Dynamic Waste Bins**: Real-time fill simulation with threshold-based notifications
- **Traffic Light Integration**: Realistic urban environment simulation

### ✅ **Advanced Q-Learning Implementation**
- **State Representation**: 6-dimensional state space (position, intersection, load, energy, target)
- **Reward System**: Sophisticated scoring for pickup, dropoff, charging, and movement efficiency
- **Adaptive Learning**: Configurable epsilon-greedy exploration with learning rate adaptation
- **Performance Tracking**: Real-time action logging and learning progress monitoring

### ✅ **Coordinate Transformation System**
- **JSON-to-Grid Mapping**: Seamless transformation from Unity world coordinates to simulation grid
- **Configurable Offsets**: Support for different coordinate systems (offset_x: 260, offset_z: 120)
- **Boundary Validation**: Robust position checking with fallback to road network
- **Multi-Scale Support**: Handles coordinate ranges from -260 to +200 mapped to 500×400 grid

### ✅ **Unity Integration & API**
- **RESTful API Server**: Flask-based server with CORS support for Unity WebGL
- **Real-Time Data Exchange**: Live simulation state and history endpoints
- **File Serving**: Static visualization and configuration file access
- **Health Monitoring**: Comprehensive server status and file availability checking

## 📁 **Project Structure**

```
TidyMesh_Sim/
├── 📄 Core Simulation
│   ├── TidyMesh_Sim_v2.py           # 🎯 Main simulation engine with Q-Learning
│   ├── requirements.txt             # 📦 Python dependencies  
│   ├── README.md                    # 📖 This documentation
│   └── TidyMesh_API_Endpoints.md    # 🌐 Complete API documentation
│
├── ⚙️ Configuration
│   └── config_Sim/
│       ├── roadZones.json           # 🛣️ Road network topology (4000+ lanes)
│       ├── trashBinZones.json       # 🗑️ Waste bin locations (60+ zones)
│       └── trafficLights.json       # 🚦 Traffic light positions & cycles
│
├── 🔧 API & Scripts  
│   └── scripts/
│       ├── tidymesh_api_server_v2.py # 🌐 Flask API server for Unity
│       ├── test_api_client.py        # 🧪 API testing utilities
│       ├── visualizer.py             # 📊 Advanced visualization engine
│       └── debug_history.py          # � Simulation analysis tools
│
├── 📊 Results & Data
│   └── results/
│       ├── simulation_data/          # 💾 JSON simulation outputs
│       │   ├── mas_final_state.json     # Final agent states
│       │   └── simulation_history.json  # Complete step history
│       └── visualizations/           # � Generated graphics
│           ├── simulation_overview.png     # Performance summary
│           ├── qlearning_analysis.png      # Learning analytics  
│           └── simulation_animation.gif    # Animated simulation
│
└── 📚 Documentation
    └── documentation/
        ├── Advanced_QLearning_Documentation.md  # 🧠 Q-Learning architecture
        └── TidyMesh_API_Unity_Guide.md         # 🎮 Unity integration guide
```

## 🎯 **Core System Components**

### **🤖 Agent Architecture**
```python
# Multi-Agent System Components
├── GarbageTruck (Base Agent)
│   ├── Q-Learning Engine (6D state space)
│   ├── Contract Net Protocol Client  
│   ├── Dynamic Pathfinding (BFS + road network)
│   └── Energy & Load Management
│
├── Dispatcher (Coordination Agent)
│   ├── Contract Net Protocol Server
│   ├── Task Assignment & Bidding
│   ├── Fairness Algorithm
│   └── Fallback Assignment
│
├── TrashBin (Environment Agent)  
│   ├── Dynamic Fill Simulation
│   ├── State Management (Ready/Servicing/Done)
│   ├── Notification System
│   └── Threshold-Based Alerts
│
├── TrafficLight (Infrastructure Agent)
│   ├── Cyclic State Management
│   ├── Coordinate Transformation
│   └── Agent Movement Control
│
└── Depot (Service Agent)
    ├── Truck Charging Station
    ├── Waste Unloading Point
    └── Fleet Management Hub
```

### **🧠 Q-Learning Implementation**
```python
# State Representation (6 dimensions)
q_state = (
    bucket_x,        # Position X (0-9 buckets)  
    bucket_y,        # Position Y (0-9 buckets)
    at_intersection, # Boolean intersection flag (0/1)
    load_level,      # Load capacity ratio (0-3)
    energy_level,    # Energy level (0-3)  
    target_type      # Target type: None/Depot/Bin (0-2)
)

# Action Space
ACTIONS = ["FORWARD", "LEFT", "RIGHT", "PICK", "DROP", "CHARGE", "WAIT"]

# Learning Parameters
alpha = 0.8      # Learning rate (configurable)
gamma = 0.95     # Discount factor
epsilon = 0.1    # Exploration rate
```

### **🌐 API Integration**
```python
# RESTful API Endpoints
GET  /                                      # API information
GET  /health                               # Health status  
GET  /status                               # Server status
GET  /TidyMesh/Sim/v2/mas_final_state.json      # Final state
GET  /TidyMesh/Sim/v2/simulation_history.json   # Complete history
POST /TidyMesh/Sim/v2/run                       # Start simulation
GET  /TidyMesh/Sim/v2/run/status                # Run status
```

## 🚀 **Quick Start Guide**

### **1. 🏃‍♂️ Run Simulation**
```bash
# Navigate to project directory
cd TidyMesh_Sim

# Run main simulation with default parameters
python TidyMesh_Sim_v2.py

# Expected Output: Real-time agent updates, bin completion tracking
```

### **2. 🌐 Start API Server (for Unity Integration)**
```bash
# Start Flask API server
cd scripts
python tidymesh_api_server_v2.py

# Server starts on http://localhost:5000
# CORS enabled for Unity WebGL builds
```

### **3. 🧪 Test API Endpoints**
```bash
# Test all endpoints and connectivity
cd scripts  
python test_api_client.py

# Validates: Server health, data availability, response times
```

### **4. 📊 Generate Visualizations**
```bash
# Create comprehensive analysis plots and animations
cd scripts
python visualizer.py

# Outputs: simulation_overview.png, qlearning_analysis.png, simulation_animation.gif
```

## ⚙️ **Configuration & Parameters**

### **🎛️ Simulation Parameters**
```python
# Core Simulation Settings
{
    "width": 500,              # Grid width (cells)
    "height": 400,             # Grid height (cells)  
    "steps": 100000,           # Maximum simulation steps
    
    # Agent Configuration
    "n_trucks": 15,            # Number of garbage trucks
    "n_bins": 10,              # Number of waste bins
    "n_tlights": 10,           # Number of traffic lights
    "n_obstacles": 8,          # Number of moving obstacles
    
    # Coordinate Transformation
    "coord_offset_x": 260,     # X-axis offset for JSON coordinates
    "coord_offset_z": 120,     # Z-axis offset for JSON coordinates
    
    # Truck Parameters
    "truck_capacity": 4.0,     # Maximum truck load capacity
    "pick_amount": 0.5,        # Amount picked per action
    "unload_threshold": 0.8,   # Load ratio to trigger depot return
    "energy_max": 200,         # Maximum truck energy
    
    # Bin Parameters  
    "bin_threshold": 0.8,      # Fill level for bin to become "Ready"
    "bin_fill_rate": 0.05,     # Rate of bin filling per step
    
    # Q-Learning Configuration
    "q_alpha": 0.8,            # Learning rate
    "q_gamma": 0.95,           # Discount factor
    "q_epsilon": 0.1,          # Exploration rate
}
```

### **🗺️ Coordinate System**
```python
# JSON World Coordinates → Simulation Grid Mapping
def transform_coordinates(json_x, json_z, offset_x=260, offset_z=120):
    """
    Transform Unity world coordinates to simulation grid
    
    Input Range:  X: [-260, +200], Z: [-120, +280] 
    Output Range: X: [0, 500],     Y: [0, 400]
    """
    grid_x = int(json_x + offset_x)  # Shift and convert
    grid_z = int(json_z + offset_z)  # Shift and convert
    return grid_x, grid_z

# Validation ensures positions stay within bounds
def is_valid_position(x, z, width=500, height=400):
    return 0 <= x < width and 0 <= z < height
```


### **🎯 Key Performance Indicators**
| Metric | Description | Target |
|--------|-------------|--------|
| **Bin Completion Rate** | Percentage of bins fully serviced | >80% |
| **Fleet Efficiency** | Distance per waste unit collected | <20 units |
| **Learning Convergence** | Q-value stabilization time | <50K steps |
| **Coordination Success** | Successful contract negotiations | >90% |
| **Resource Utilization** | Truck capacity and energy usage | >75% |

### **🔬 Q-Learning Analytics**
```python
# Learning Progress Indicators
{
    "q_table_size": 1500,         # Number of learned state-action pairs
    "exploration_rate": 0.1,      # Current epsilon value
    "learning_rate": 0.8,         # Current alpha value  
    "reward_convergence": True,   # Whether rewards are stabilizing
    "action_distribution": {      # Action usage statistics
        "FORWARD": 0.45,
        "LEFT": 0.15, 
        "RIGHT": 0.15,
        "PICK": 0.10,
        "DROP": 0.05,
        "CHARGE": 0.05,
        "WAIT": 0.05
    }
}
```

## 🛠 **Technical Implementation**

### **🏗️ System Architecture**
```python
# Core Framework: AgentPy (Mesa-compatible)
# Backend: Python 3.8+
# API: Flask + CORS for Unity integration
# Pathfinding: BFS with road network graphs
# Learning: Q-Learning with epsilon-greedy exploration
# Coordination: Contract Net Protocol
# Visualization: Matplotlib + Seaborn + PIL

# Key Classes & Responsibilities
├── CityWasteV2 (ap.Model)          # Main simulation model
├── Truck (Base Agent)              # Q-Learning garbage trucks  
├── Dispatcher (Base Agent)         # Task coordination hub
├── TrashBin (Base Agent)           # Dynamic waste containers
├── TrafficLight (Base Agent)       # Traffic control points
├── Depot (Base Agent)              # Service station
└── TidyMeshVisualizer              # Analytics & visualization
```

### **📡 Unity Integration Architecture** 
```python
# API Server Features
✅ RESTful API with JSON responses
✅ CORS enabled for Unity WebGL
✅ Real-time simulation data access  
✅ File serving for configurations
✅ Health monitoring & status checks
✅ Thread-safe simulation execution
✅ Caching for improved performance

# Unity Integration Flow
Unity → HTTP GET → Flask API → JSON Response → Unity JsonUtility
      ← Real-time ← Simulation ← Data Processing ←
```

### **🧠 Q-Learning Algorithm**
```python
# Standard Q-Learning Update Rule
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

# State-Action Value Update
def update_q_table(self, state, action, reward, next_state):
    current_q = self.Q[state][action]
    max_next_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0
    new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    self.Q[state][action] = new_q

# Reward Function
reward = base_movement_cost +              # -1 for movement
         pickup_bonus +                    # +10 for successful pickup
         dropoff_bonus +                   # +15 for successful dropoff  
         charging_bonus +                  # +5 for charging
         efficiency_multiplier             # +capacity_ratio bonus
```

## 🎮 **Usage Examples**

### **🔧 Custom Simulation Configuration**
```python
# Create and run simulation with custom parameters
from TidyMesh_Sim_v2 import CityWasteV2

# Define custom parameters
custom_params = {
    "n_trucks": 8,              # More trucks for faster collection
    "n_bins": 20,               # More bins for longer simulation
    "steps": 50000,             # Shorter run for testing
    "q_alpha": 0.9,             # Higher learning rate
    "q_epsilon": 0.05,          # Less exploration
    "truck_capacity": 5.0,      # Larger truck capacity
}

# Run simulation
model = CityWasteV2(custom_params)
results = model.run()

# Access final statistics
print(f"Bins completed: {results['total_bins_done']}")
print(f"Total distance: {results['total_distance']}")
```

### **📊 Advanced Visualization**
```python
# Generate comprehensive analysis plots
from scripts.visualizer import TidyMeshVisualizer

viz = TidyMeshVisualizer()

# Create all visualizations
viz.create_static_overview()        # Performance summary
viz.create_qlearning_analysis()     # Learning progress analysis
viz.create_animated_simulation()    # Full simulation animation

# Custom visualization with specific parameters
viz.create_agent_trajectory_plot(truck_id="ID_101")
viz.create_efficiency_heatmap(time_window=(1000, 5000))
```

### **🌐 API Integration Example**
```python
# Unity C# integration example
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class TidyMeshAPIClient : MonoBehaviour 
{
    private string apiBaseUrl = "http://localhost:5000";
    
    // Get simulation final state
    public IEnumerator GetSimulationState() 
    {
        string url = $"{apiBaseUrl}/TidyMesh/Sim/v2/mas_final_state.json";
        
        using (UnityWebRequest request = UnityWebRequest.Get(url)) 
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success) 
            {
                SimulationState state = JsonUtility.FromJson<SimulationState>(request.downloadHandler.text);
                UpdateSimulationVisualization(state);
            }
        }
    }
}
```

## 🔧 **Dependencies & Installation**

### **📦 Required Packages**
```bash
# Install all dependencies
pip install -r requirements.txt

# Core dependencies:
pip install agentpy          # Multi-agent simulation framework
pip install numpy           # Numerical computation  
pip install pandas          # Data analysis and manipulation
pip install matplotlib      # Plotting and visualization
pip install seaborn         # Statistical data visualization
pip install pillow          # Image processing for animations
pip install flask           # Web framework for API server
pip install flask-cors      # Cross-origin resource sharing
pip install requests        # HTTP library for API testing
```

### **⚙️ System Requirements**
```
Python: 3.8 or higher
RAM: 4GB minimum (8GB recommended for large simulations)
CPU: Multi-core recommended for parallel agent processing
Storage: 500MB for simulation outputs and visualizations
Network: Port 5000 available for API server
```

### **🔐 Environment Setup**
```bash
# Create virtual environment (recommended)
python -m venv tidymesh_env

# Activate environment
# Windows:
tidymesh_env\Scripts\activate
# macOS/Linux:
source tidymesh_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import agentpy; print('AgentPy version:', agentpy.__version__)"
```

## 🏆 **Key Achievements & Features**

### ✅ **Advanced Multi-Agent Coordination**
- **Contract Net Protocol**: Distributed task allocation with competitive bidding
- **Dynamic Load Balancing**: Automatic workload distribution among available trucks
- **Intelligent Pathfinding**: BFS-based navigation with road network optimization
- **Real-Time Communication**: Agent-to-agent messaging and coordination

### ✅ **Sophisticated Q-Learning Implementation**  
- **Adaptive Learning Rates**: Dynamic adjustment based on performance
- **State Space Optimization**: 6-dimensional representation for efficient learning
- **Reward Engineering**: Carefully tuned incentive structure for optimal behavior
- **Exploration-Exploitation Balance**: Epsilon-greedy with learning progression

### ✅ **Seamless Unity Integration**
- **RESTful API Architecture**: Production-ready Flask server with comprehensive endpoints
- **Real-Time Data Streaming**: Live simulation state updates for Unity consumption
- **CORS Support**: Full WebGL compatibility for browser-based Unity applications
- **Health Monitoring**: Robust error handling and server status reporting

### ✅ **Comprehensive Visualization System**
- **Multi-Panel Analytics**: Performance metrics, learning curves, spatial analysis
- **Animated Simulations**: Real-time agent movement and interaction visualization
- **Export Capabilities**: High-quality PNG, GIF, and interactive plot generation
- **Customizable Views**: Configurable visualization parameters and styling

### ✅ **Production-Ready Coordinate System**
- **Universal Coordinate Transformation**: Seamless JSON-to-grid mapping
- **Boundary Validation**: Robust position checking with fallback mechanisms
- **Multi-Scale Support**: Handles diverse coordinate ranges and grid sizes
- **Error-Free Operation**: Eliminated coordinate-related warnings and issues

## 🌐 **API Documentation**

### **🚀 Quick API Server Setup**
```bash
# Start the API server for Unity integration
cd scripts
python tidymesh_api_server_v2.py

# Server starts on http://localhost:5000
# Comprehensive logging and status monitoring included
```

### **� Complete API Endpoints**

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|-----------------|
| `/` | GET | API information & documentation | JSON metadata |
| `/health` | GET | Server health & file status | JSON health data |
| `/status` | GET | Detailed server status | JSON status info |
| `/TidyMesh/Sim/v2/mas_final_state.json` | GET/POST | Final simulation state | Complete agent states |
| `/TidyMesh/Sim/v2/simulation_history.json` | GET/POST | Step-by-step history | Temporal simulation data |
| `/TidyMesh/Sim/v2/run` | POST | Start new simulation | Run status & parameters |
| `/TidyMesh/Sim/v2/run/status` | GET | Current run status | Execution progress |
| `/TidyMesh/Sim/v2/config/<file>` | GET | Configuration files | JSON config data |
| `/TidyMesh/Sim/v2/visualization/<file>` | GET | Visualization files | Images/animations |

### **🔗 Unity Integration Features**
```csharp
// Unity integration capabilities:
✅ CORS Enabled           - Full Unity WebGL support
✅ JSON Format            - Direct Unity JsonUtility compatibility  
✅ Real-Time Data         - Live simulation data access
✅ File Serving           - Static asset delivery
✅ Error Handling         - Comprehensive status codes
✅ Caching System         - Performance optimization
✅ Thread Safety          - Concurrent request handling
```

### **📋 API Usage Examples**
```bash
# Test server health
curl http://localhost:5000/health

# Get final simulation state  
curl http://localhost:5000/TidyMesh/Sim/v2/mas_final_state.json

# Start new simulation with parameters
curl -X POST http://localhost:5000/TidyMesh/Sim/v2/run \
  -H "Content-Type: application/json" \
  -d '{"n_trucks": 8, "n_bins": 15, "steps": 50000}'

# Check simulation status
curl http://localhost:5000/TidyMesh/Sim/v2/run/status
```

**📚 Complete Integration Guide:** [`documentation/TidyMesh_API_Unity_Guide.md`](documentation/TidyMesh_API_Unity_Guide.md)

## � **Troubleshooting & Support**

### **🚨 Common Issues & Solutions**

#### **Simulation Performance**
```python
# Issue: Slow simulation execution
# Solution: Reduce simulation complexity
params = {
    "n_trucks": 5,        # Reduce from default 15
    "n_bins": 8,          # Reduce from default 10  
    "steps": 25000,       # Reduce from default 100000
    "n_obstacles": 3      # Reduce moving obstacles
}
```

#### **API Connection Issues**
```bash
# Issue: Unity can't connect to API
# Solution: Check server status and port availability

# 1. Verify server is running
curl http://localhost:5000/health

# 2. Check port availability  
netstat -an | findstr :5000

# 3. Restart server with debug mode
python tidymesh_api_server_v2.py --debug --port 5000
```

#### **Coordinate Transformation Problems**
```python
# Issue: Agents appearing outside grid bounds
# Solution: Verify coordinate transformation parameters

# Check current offsets match your data
coord_offset_x = 260  # For JSON X range [-260, +200]
coord_offset_z = 120  # For JSON Z range [-120, +280]

# Validate positions before use
def validate_position(x, z, width=500, height=400):
    if not (0 <= x < width and 0 <= z < height):
        print(f"Warning: Position ({x}, {z}) outside bounds")
        return False
    return True
```

### **💡 Performance Optimization Tips**
- Use smaller grid sizes for faster execution
- Reduce the number of agents for initial testing
- Enable visualization only when needed (affects performance)
- Monitor memory usage with large simulation datasets
- Use background mode for long-running simulations

### **📞 Support Resources**
- **Documentation**: `documentation/` folder for detailed guides
- **API Reference**: `TidyMesh_API_Endpoints.md` for complete endpoint list
- **Code Examples**: `scripts/test_api_client.py` for API usage examples
- **Debugging Tools**: `scripts/debug_history.py` for simulation analysis

---

## 📝 **License & Academic Information**

### **📚 Academic Project**
This is an academic project for the **Multi-Agent Computer Graphics** course at Universidad.

### **🎓 Course Information**
- **Institution**: Instituto Tecnologico y de Estudios Superiores de Monterrey  
- **Semester**: 5th Semester (Quinto Semestre)
- **Course**: Multi-Agent Computer Graphics (MultiAgentsComputerGraphics)
- **Project**: Advanced Multi-Agent Waste Collection Simulation
- **Development Period**: 2025

### **👥 Development Team**
- **Santiago Quintana Moreno** - A01571222
- **Sergio Rodríguez Pérez** - A00838856  
- **Rodrigo González de la Garza** - A00838952
- **Diego Gaitan Sanchez** - A01285960
- **Miguel Ángel Álvarez Hermida** - A01722925

### **🏢 Project Sponsor**
**NDS Cognitive Labs Mexico**

### **📄 Copyright Notice**
```
COPYRIGHT 2025 TIDYMESH INC. ALL RIGHTS RESERVED.

This project is developed for academic purposes as part of the 
Multi-Agent Computer Graphics course curriculum. 

All code, documentation, and associated materials are protected 
under academic use guidelines and copyright law.
```

### **📈 Version History**
- **v1.0** - Initial multi-agent simulation implementation
- **v2.0** - Enhanced Q-Learning, API integration, Unity support
- **Current** - Production-ready simulation with comprehensive documentation

---

## 🚀 **Get Started Today!**

```bash
# Clone and run in 3 simple steps:
git clone <repository-url>
cd TidyMesh_Sim  
python TidyMesh_Sim_v2.py

# 🎉 Watch intelligent agents learn and collaborate!
```

**For questions, issues, or contributions, please refer to the comprehensive documentation in the `/documentation` folder.**

### **📞 Support Resources**
- **Documentation**: `documentation/` folder for detailed guides
- **API Reference**: `TidyMesh_API_Endpoints.md` for complete endpoint list
- **Code Examples**: `scripts/test_api_client.py` for API usage examples
- **Debugging Tools**: `scripts/debug_history.py` for simulation analysis
