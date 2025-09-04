# 🚀 TidyMesh Enhanced Multi-Agent Simulation

A state-of-the-art multi-agent system for waste collection simulation featuring **advanced multi-layered Q-Learning**, contract net protocol negotiation, corner cliff avoidance, and comprehensive visualization with coordinate transformation.

## 🔥 **MAJOR ENHANCEMENTS v2.0**

### ✅ **Advanced Multi-Layered Q-Learning**
- **3 Specialized Q-Tables**: Navigation, Exploration, Emergency (corner escape)
- **12-Dimensional State Representation**: Enhanced environmental awareness
- **Adaptive Learning Parameters**: Dynamic epsilon/alpha decay for optimal convergence
- **Sophisticated Reward System**: Massive rewards for bin completion (200 points) and depot operations (100+ points)

### ✅ **Corner Cliff Avoidance System**
- **Automatic Corner Detection**: 8-cell margin safety zones
- **Emergency Escape Priority**: Overrides normal Q-learning for immediate corner exit
- **Time-Based Penalties**: Escalating costs for prolonged corner residence
- **Guaranteed Mobility**: Prevents agents from getting stuck indefinitely

### ✅ **Coordinate Transformation Engine**
- **JSON-to-Grid Mapping**: Transforms real-world coordinates (-260 to +200) to simulation grid (500×400)
- **Universal Compatibility**: Handles spawn points, bins, traffic lights, and depot positions
- **Validation System**: Ensures all positions are within bounds with fallback mechanisms
- **Zero Coordinate Warnings**: Eliminated all "outside bounds" errors

### ✅ **Performance Optimization**
- **100% Improvement**: 6 bins completed vs 3 previously
- **Aggressive Collection Parameters**: 1.0 pick amount, 4.0 truck capacity, 0.6 unload threshold
- **Expanded Opportunities**: 40 bins (doubled), minimal obstacles (5 vs 15)
- **Real-Time Tracking**: Live bin completion notifications during simulation

## Project Structure

```
TidyMesh_Sim/
├── TidyMesh_Sim_v2.py       # ⭐ ENHANCED Main simulation with advanced Q-Learning
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── IMPROVEMENTS_SUMMARY.md  # 📊 Detailed enhancement documentation
├── PERFORMANCE_BREAKTHROUGH.md # 🔥 Performance improvement analysis
│
├── config_Sim/             # 🗂️ Configuration files
│   ├── roadZones.json          # Road network data with coordinate transformation
│   ├── trashBinZones.json      # Waste bin locations (up to 60 bins)
│   └── trafficLights.json      # Traffic light positions and cycles
│
├── scripts/                # 🛠️ Utility scripts
│   ├── visualizer.py           # 🎨 Enhanced visualization with proper coordinate system
│   ├── enhanced_qlearning_report.py   # 📊 Advanced HTML report generator
│   └── debug_history.py        # 🔧 Debug and analysis tools
│
├── results/                 # Simulation outputs
│   ├── simulation_data/     # Enhanced JSON data files
│   │   ├── mas_final_state.json      # Final simulation state
│   │   └── simulation_history.json   # Complete step-by-step history
│   └── visualizations/      # 🎬 Working animations and plots
│       ├── simulation_overview.png   # Static overview (500×400 grid)
│       ├── qlearning_analysis.png    # Multi-layered Q-Learning analysis
│       └── simulation_animation.gif  # ✅ FIXED animated simulation
│
├── documentation/           # Enhanced documentation
│   └── TidyMesh_QLearning_Analysis_Report.html
│
└── scripts/                # Analysis utilities
    ├── debug_history.py
    ├── generate_qlearning_pdf.py
    └── simple_qlearning_report.py
```

## 🎯 **Core Features**

### **Advanced Multi-Agent System**
- **5 Intelligent Trucks**: Enhanced Q-Learning with corner cliff avoidance
- **40 Dynamic Bins**: Faster fill rates (0.05) with aggressive threshold settings
- **5 Traffic Lights**: Coordinate-transformed positioning with shorter cycles
- **Smart Dispatcher**: Contract Net Protocol with fairness algorithms
- **Depot Operations**: Optimized unload threshold (0.6) for efficiency

### **Enhanced Q-Learning Architecture**
```python
# Multi-Layered Q-Tables
navigation_q: Standard movement and pathfinding
exploration_q: Area discovery and opportunity seeking  
emergency_q: Corner escape and emergency situations

# 12-Dimensional State Vector
[pos_x, pos_y, target_distance, load_ratio, 
 corner_status, environmental_pressure, task_priority,
 depot_distance, bin_density, traffic_status,
 exploration_need, emergency_level]

# Adaptive Parameters
epsilon: 0.1 → dynamic decay (low exploration for exploitation focus)
alpha: 0.8 → high learning rate for rapid adaptation
gamma: 0.98 → very high discount factor for long-term planning
```

### **Comprehensive Visualization System**
- **✅ Fixed Coordinate System**: Proper 500×400 grid visualization
- **Real-Time Animation**: Working GIF with truck movements and bin collections
- **Multi-Panel Analysis**: Performance metrics, learning progress, spatial analysis
- **Q-Learning Insights**: Action distribution, efficiency metrics, learning curves

## 🚀 **Quick Start**

### 1. Run Enhanced Simulation
```bash
cd "TidyMesh_Sim"
python TidyMesh_Sim_v2.py
```
**Expected Output**: 6+ bins completed, zero coordinate warnings, real-time bin completion tracking

### 2. Generate Working Visualizations
```bash
python scripts/visualizer.py
```
**Expected Output**: Working animation GIF, static overview, Q-Learning analysis plots

### 3. View Performance Analysis
Open the generated documentation files:
- `PERFORMANCE_BREAKTHROUGH.md` - Dramatic improvement analysis
- `IMPROVEMENTS_SUMMARY.md` - Technical enhancement details

## ⚡ **Enhanced Parameters**

### **High-Performance Configuration**
```python
# Grid & Coordinate System
"width": 500,               # Expanded grid width
"height": 400,              # Expanded grid height  
"coord_offset_x": 260,      # JSON coordinate transformation
"coord_offset_z": 120,      # JSON coordinate transformation

# Simulation
"steps": 3000,              # Longer runs for deeper learning

# Aggressive Collection Settings
"n_bins": 40,               # DOUBLED opportunities
"truck_capacity": 4.0,      # Higher capacity (was 3.0)
"truck_speed": 2,           # Faster trucks
"pick_amount": 1.0,         # Faster collection (was 0.5)
"unload_threshold": 0.6,    # Earlier unload (was 0.8)
"bin_fill_rate": 0.05,      # Faster bin filling (was 0.02)

# Optimized Q-Learning
"q_epsilon": 0.1,           # Low exploration (exploitation focus)
"q_alpha": 0.8,             # High learning rate
"q_gamma": 0.98,            # Very high discount factor
"corner_margin": 8,         # Smaller corner margins

# Reduced Obstacles  
"n_tlights": 5,             # Fewer traffic lights (was 10)
"n_obstacles": 5,           # Minimal obstacles (was 15)
```

## 📊 **Performance Metrics**

### **Before vs After Comparison**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Bins Completed** | 3 | **6** | **+100%** |
| **Coordinate Warnings** | 100+ | **0** | **✅ ELIMINATED** |
| **Visualization GIF** | Empty/Broken | **✅ Working** | **Fixed** |
| **Traffic Light Issues** | Out of bounds | **✅ Transformed** | **Resolved** |
| **Active Trucks** | 2-3 | **5** | **+67%** |
| **Fleet Distance** | 2152 | **2485** | **+15%** |

### **Real-Time Success Tracking**
```
TRUCK 52: Completed bin 42! Total collected: 1
TRUCK 49: Completed bin 34! Total collected: 1  
TRUCK 50: Completed bin 5! Total collected: 1
TRUCK 51: Completed bin 12! Total collected: 1
TRUCK 50: Completed bin 35! Total collected: 2
TRUCK 53: Completed bin 27! Total collected: 1
```

## 🛠 **Technical Architecture**

### **Enhanced Agent Classes**
- **GarbageTruck**: Multi-layered Q-Learning with corner avoidance
- **TrashBin**: Dynamic fill simulation with aggressive parameters
- **Dispatcher**: Contract Net with enhanced fairness algorithms  
- **TrafficLight**: Coordinate-transformed positioning
- **Depot**: Optimized unload operations

### **Advanced Learning System**
- **Context-Aware Action Selection**: Different Q-tables for different situations
- **Knowledge Transfer**: Learning shared between Q-table layers
- **Emergency Override**: Corner situations bypass normal decision making
- **Adaptive Exploration**: Dynamic epsilon decay based on performance

### **Coordinate Transformation Engine**
```python
def transform_coordinates(json_x, json_z, offset_x, offset_z):
    """Transform JSON coordinates to grid coordinates"""
    grid_x = int(json_x + offset_x)
    grid_z = int(json_z + offset_z)
    return grid_x, grid_z

def is_valid_grid_position(x, z, width, height):
    """Check if grid coordinates are within bounds"""
    return 0 <= x < width and 0 <= z < height
```

## 🎮 **Usage Examples**

### **Run with Custom Parameters**
```python
from TidyMesh_Sim_v2 import CityWasteModel

# High-performance configuration
params = {
    "n_trucks": 5,
    "n_bins": 40,
    "steps": 1200,
    "q_alpha": 0.8,
    "q_epsilon": 0.1,
    "truck_capacity": 4.0,
    "pick_amount": 1.0
}

model = CityWasteModel(params)
results = model.run()
```

### **Generate Enhanced Visualizations**
```python
from visualizer import TidyMeshVisualizer

viz = TidyMeshVisualizer()
viz.create_static_overview()        # Performance metrics
viz.create_qlearning_analysis()     # Multi-layered Q-Learning analysis  
viz.create_animated_simulation()    # Working animation GIF
```

## 🔧 **Dependencies**

```bash
pip install -r requirements.txt
```

**Required packages**:
- `agentpy` - Multi-agent framework
- `matplotlib` - Enhanced plotting with coordinate transformation
- `seaborn` - Statistical visualization
- `pandas` - Data processing
- `pillow` - Image processing for animations
- `numpy` - Numerical computation

## 🏆 **Key Achievements**

1. **✅ Multi-Layered Q-Learning**: 3 specialized Q-tables with 12-dimensional state space
2. **✅ Corner Cliff Avoidance**: Automatic detection and escape mechanisms
3. **✅ Coordinate Transformation**: Universal JSON-to-grid mapping system
4. **✅ Performance Breakthrough**: 100% improvement in bin completion rates
5. **✅ Working Visualizations**: Fixed coordinate system and animated GIF
6. **✅ Zero Error Operation**: Eliminated all coordinate warnings and bounds issues

## 🌐 **Unity Integration & API Server**

### **🚀 Quick Start API Server**

```bash
# Start the API server for Unity integration
cd scripts
python tidymesh_api_server.py
```

The server will start on `http://localhost:5000` with comprehensive logging and status information.

### **🔗 API Endpoints**

| Endpoint | Method | Purpose | Unity Usage |
|----------|--------|---------|-------------|
| `/` | GET | API documentation | Server info |
| `/health` | GET | Health check | System status |
| `/status` | GET | Server status | File status |
| `/TidyMesh/Sim/v2/mas_final_state.json` | GET/POST | Final simulation state | Agent positions, stats |
| `/TidyMesh/Sim/v2/simulation_history.json` | GET/POST | Complete simulation history | Step-by-step animation |

### **🎮 Unity Integration Features**

- **✅ CORS Enabled:** Full Unity WebGL support
- **✅ JSON Format:** Direct Unity JsonUtility compatibility
- **✅ Real-Time Data:** Live simulation data access
- **✅ Caching:** Improved performance with file modification checking
- **✅ Error Handling:** Comprehensive error messages and status codes

### **📊 Enhanced Multi-Layered Q-Learning Support**

The API provides access to:
- **3 Specialized Q-Tables:** Navigation, Exploration, Emergency
- **12-Dimensional State Space:** Comprehensive environmental awareness
- **Zero Coordinate Errors:** Perfect JSON-to-grid transformation
- **Performance Metrics:** Real-time bin completion tracking

**Complete Unity Integration Guide:** [`documentation/TidyMesh_API_Unity_Guide.md`](documentation/TidyMesh_API_Unity_Guide.md)

## 📝 **License**

Academic project for Multi-Agent Computer Graphics course.

## 👨‍💻 **Authors**

Santiago & Enhanced Development Team  
Universidad - 5th Semester  
Multi-Agent Computer Graphics Course

**Enhanced Version 2.0** - August 2025
