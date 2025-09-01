# 🌐 **TidyMesh API Server & Unity Integration Guide**

**Date:** August 31, 2025  
**Version:** TidyMesh v2.0 Enhanced Multi-Layered Q-Learning API

---

## 🚀 **Quick Start**

### **1. Start the API Server**
```bash
cd TidyMesh_Sim/scripts
python tidymesh_api_server.py
```

### **2. Test the API**
```bash
# In a new terminal
cd TidyMesh_Sim/scripts  
python test_api_client.py
```

### **3. Access from Unity**
Use the endpoints with UnityWebRequest to fetch simulation data.

---

## 🔗 **API Endpoints**

### **Base URL:** `http://localhost:5000`

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/` | GET | API information and documentation | API metadata |
| `/health` | GET | Health check and file status | System status |
| `/status` | GET | Detailed server status | Cache and file info |
| `/TidyMesh/Sim/v2/mas_final_state.json` | GET/POST | Final simulation state | Complete final state |
| `/TidyMesh/Sim/v2/simulation_history.json` | GET/POST | Complete simulation history | Step-by-step data |

### **Additional Endpoints**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/TidyMesh/Sim/v2/visualization/<filename>` | GET | Serve visualization files |
| `/TidyMesh/Sim/v2/config/<filename>` | GET | Serve configuration files |

---

## 🎮 **Unity Integration**

### **1. UnityWebRequest GET Example** 
```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class TidyMeshAPIClient : MonoBehaviour
{
    private const string API_BASE = "http://localhost:5000";
    private const string FINAL_STATE_URL = API_BASE + "/TidyMesh/Sim/v2/mas_final_state.json";
    private const string HISTORY_URL = API_BASE + "/TidyMesh/Sim/v2/simulation_history.json";

    [System.Serializable]
    public class APIResponse
    {
        public string timestamp;
        public string source;
        public string version;
        public SimulationData data;
    }

    [System.Serializable]
    public class SimulationData
    {
        public Agent[] agents;
        public int total_steps;
        public string simulation_id;
    }

    [System.Serializable]
    public class Agent
    {
        public string id;
        public string type;
        public float[] position;
        public float load;
        public int bins_completed;
    }

    public IEnumerator GetFinalState()
    {
        using (UnityWebRequest request = UnityWebRequest.Get(FINAL_STATE_URL))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string jsonResponse = request.downloadHandler.text;
                APIResponse response = JsonUtility.FromJson<APIResponse>(jsonResponse);
                
                Debug.Log($"Received simulation data: {response.data.agents.Length} agents");
                
                // Process the agents data for visualization
                ProcessAgents(response.data.agents);
            }
            else
            {
                Debug.LogError($"API Error: {request.error}");
            }
        }
    }

    public IEnumerator GetSimulationHistory()
    {
        using (UnityWebRequest request = UnityWebRequest.Get(HISTORY_URL))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string jsonResponse = request.downloadHandler.text;
                APIResponse response = JsonUtility.FromJson<APIResponse>(jsonResponse);
                
                Debug.Log($"Received history data from: {response.source}");
                
                // Process historical data for timeline visualization
                ProcessHistory(response.data);
            }
        }
    }

    private void ProcessAgents(Agent[] agents)
    {
        foreach (Agent agent in agents)
        {
            if (agent.type == "GarbageTruck")
            {
                // Update truck position and status
                Vector3 position = new Vector3(agent.position[0], 0, agent.position[1]);
                UpdateTruckVisualization(agent.id, position, agent.load, agent.bins_completed);
            }
            else if (agent.type == "TrashBin")
            {
                // Update bin status
                Vector3 position = new Vector3(agent.position[0], 0, agent.position[1]);
                UpdateBinVisualization(agent.id, position);
            }
        }
    }

    private void ProcessHistory(SimulationData data)
    {
        // Process step-by-step simulation data for animation
        Debug.Log($"Processing {data.total_steps} simulation steps");
    }

    private void UpdateTruckVisualization(string id, Vector3 position, float load, int bins)
    {
        // Your truck visualization logic here
        Debug.Log($"Truck {id}: Position {position}, Load {load}, Bins {bins}");
    }

    private void UpdateBinVisualization(string id, Vector3 position)
    {
        // Your bin visualization logic here
        Debug.Log($"Bin {id}: Position {position}");
    }
}
```

### **2. UnityWebRequest POST Example**
```csharp
public IEnumerator PostSimulationData(SimulationData data)
{
    string jsonData = JsonUtility.ToJson(data);
    byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);

    using (UnityWebRequest request = new UnityWebRequest(FINAL_STATE_URL, "POST"))
    {
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Simulation data posted successfully");
        }
        else
        {
            Debug.LogError($"POST Error: {request.error}");
        }
    }
}
```

---

## 📊 **Response Format**

### **Final State Response**
```json
{
    "timestamp": "2025-08-31T20:45:00",
    "source": "TidyMesh Enhanced Multi-Layered Q-Learning Simulation",
    "version": "v2",
    "data": {
        "simulation_id": "sim_20250831_204500",
        "total_steps": 1200,
        "agents": [
            {
                "id": "TRUCK_50",
                "type": "GarbageTruck", 
                "position": [250, 200],
                "load": 2.5,
                "capacity": 4.0,
                "bins_completed": 3,
                "total_distance": 450.7,
                "q_learning_stats": {
                    "navigation_q_size": 1250,
                    "exploration_q_size": 890,
                    "emergency_q_size": 340
                }
            },
            {
                "id": "BIN_001",
                "type": "TrashBin",
                "position": [100, 150],
                "fill_level": 0.0,
                "status": "done",
                "volume_collected": 3.2
            }
        ],
        "performance": {
            "bins_completed": 6,
            "total_distance": 2485.3,
            "simulation_time_seconds": 60.5,
            "coordinate_warnings": 0
        }
    }
}
```

### **Simulation History Response**
```json
{
    "timestamp": "2025-08-31T20:45:00",
    "source": "TidyMesh Enhanced Multi-Layered Q-Learning Simulation",
    "version": "v2", 
    "data": {
        "simulation_id": "sim_20250831_204500",
        "total_steps": 1200,
        "steps": [
            {
                "step": 1,
                "timestamp": 0.1,
                "agents": [
                    {
                        "id": "TRUCK_50",
                        "position": [260, 120],
                        "action": "MOVE_NORTH",
                        "reward": -0.1
                    }
                ]
            }
        ]
    }
}
```

---

## ⚙️ **Server Configuration**

### **Command Line Options**
```bash
python tidymesh_api_server.py --help

# Custom host and port
python tidymesh_api_server.py --host 0.0.0.0 --port 8080

# Debug mode
python tidymesh_api_server.py --debug
```

### **Environment Variables**
```bash
# Set custom configuration (optional)
export TIDYMESH_API_HOST=localhost
export TIDYMESH_API_PORT=5000
export TIDYMESH_DEBUG=false
```

---

## 🔧 **Features**

### **✅ Enhanced Multi-Layered Q-Learning Support**
- **3 Specialized Q-Tables:** Navigation, Exploration, Emergency
- **12-Dimensional State Space:** Comprehensive environmental awareness
- **Corner Cliff Avoidance:** Automatic detection and escape
- **Coordinate Transformation:** JSON-to-grid mapping with zero errors

### **✅ Real-Time Data Access**
- **File Caching:** Improved performance with modification-time checking
- **CORS Enabled:** Full Unity WebGL support
- **JSON Format:** Direct Unity JsonUtility compatibility
- **Error Handling:** Comprehensive error messages and status codes

### **✅ Development Features**
- **Health Monitoring:** File existence and size checking
- **Status Endpoints:** Detailed server and cache information
- **Test Client:** Complete integration testing
- **Logging:** Detailed request and error logging

---

## 📁 **File Structure**

```
TidyMesh_Sim/
├── scripts/
│   ├── tidymesh_api_server.py     # 🌐 Main API server
│   ├── test_api_client.py         # 🧪 API testing client
│   └── visualizer.py              # 🎨 Visualization tools
├── config_Sim/                   # 🗂️ Configuration files
├── results/
│   ├── simulation_data/           # 📊 JSON data served by API
│   └── visualizations/            # 🎬 Visualization files
└── documentation/
    └── TidyMesh_API_Unity_Guide.md # 📚 This guide
```

---

## 🚀 **Usage Workflow**

### **For Unity Developers:**

1. **Start API Server**
   ```bash
   python tidymesh_api_server.py
   ```

2. **Run TidyMesh Simulation**
   ```bash
   python TidyMesh_Sim_v2.py
   ```

3. **Fetch Data in Unity**
   ```csharp
   StartCoroutine(GetFinalState());
   StartCoroutine(GetSimulationHistory());
   ```

4. **Process and Visualize**
   - Parse JSON responses
   - Update Unity GameObjects
   - Animate based on historical data

### **For Python Developers:**

1. **Extend API Endpoints**
   - Add custom routes in `tidymesh_api_server.py`
   - Implement new data processing functions

2. **Integration with Simulation**
   - Use POST endpoints to send real-time data
   - Implement WebSocket for live updates (future)

---

## 🎯 **Benefits for Unity**

### **🎮 Direct Integration**
- **No File Watching:** Unity doesn't need to monitor file changes
- **HTTP Standard:** Use familiar UnityWebRequest patterns
- **Real-Time:** Get latest simulation data instantly
- **Cross-Platform:** Works with Unity Editor, Standalone, and WebGL

### **📊 Enhanced Data Access**
- **Multi-Layered Q-Learning:** Access all three Q-table types
- **Performance Metrics:** Real-time bin completion tracking
- **Coordinate Accuracy:** Zero coordinate transformation errors
- **Complete History:** Step-by-step simulation playback

### **🔧 Development Benefits**
- **Easy Testing:** Built-in test client and status endpoints
- **Error Handling:** Clear error messages and status codes
- **Caching:** Improved performance for repeated requests
- **Extensible:** Easy to add new endpoints and features

---

## ✨ **API Server Ready for Unity Integration!**

The TidyMesh API Server provides a professional, RESTful interface for Unity to access the enhanced multi-layered Q-Learning simulation data. With built-in caching, error handling, and Unity-specific response formats, it's ready for production visualization projects.

**🎬 Perfect for Unity visualization of the breakthrough 6-bin performance with zero coordinate errors!**
