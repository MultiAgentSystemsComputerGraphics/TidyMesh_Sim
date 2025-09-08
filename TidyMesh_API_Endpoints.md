# TidyMesh Simulation API - Complete Endpoints Documentation

## Server Information
- **API Version**: v3
- **Base Path**: `/TidyMesh/Sim`
- **Default Host**: `localhost`
- **Default Port**: `5000`
- **Base URL**: `http://localhost:5000`

## Complete Endpoint URLs

### 1. Root/Info Endpoints

#### API Information
- **URL**: `http://localhost:5000/`
- **Method**: `GET`
- **Description**: Returns API information, version, and available endpoints
- **Response**: JSON with API metadata and endpoint list

#### Health Check
- **URL**: `http://localhost:5000/health`
- **Method**: `GET`
- **Description**: Health status of the API and file availability
- **Response**: JSON with health status and file information

#### API Status
- **URL**: `http://localhost:5000/status`
- **Method**: `GET`
- **Description**: Detailed API status including cache information and file details
- **Response**: JSON with comprehensive status information

### 2. Simulation Control Endpoints

#### Start Simulation Run
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/run`
- **Method**: `POST`
- **Description**: Starts a new simulation run with configurable parameters
- **Body Parameters** (JSON):
  ```json
  {
    "n_trucks": 5,      // Number of trucks (0-50)
    "n_bins": 40,       // Number of bins (0-1000)
    "n_tlights": 10,    // Number of traffic lights (0-200)
    "n_obstacles": 0,   // Number of obstacles (0-200)
    "steps": 1200       // Simulation steps (10-2000000)
  }
  ```
- **Response**: JSON with run status and result URLs

#### Simulation Run Status
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/run/status`
- **Method**: `GET`
- **Description**: Returns the current status of simulation runs
- **Response**: JSON with run state, timestamps, parameters, and error information

### 3. Data Endpoints

#### Final State Data
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/mas_final_state.json`
- **Methods**: `GET`, `POST`
- **Description**: 
  - `GET`: Retrieves the final state of the last simulation
  - `POST`: Saves final state data to the server
- **Response**: JSON with final simulation state data

#### Simulation History Data
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/simulation_history.json`
- **Methods**: `GET`, `POST`
- **Description**: 
  - `GET`: Retrieves the complete simulation history
  - `POST`: Saves simulation history data to the server
- **Response**: JSON with step-by-step simulation history

### 4. Static File Endpoints

#### Visualization Files
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/visualization/<filename>`
- **Method**: `GET`
- **Description**: Serves visualization files (images, animations, etc.)
- **Parameters**: 
  - `<filename>`: Name of the visualization file
- **Examples**:
  - `http://localhost:5000/TidyMesh/Sim/v2/visualization/simulation_animation.gif`
  - `http://localhost:5000/TidyMesh/Sim/v2/visualization/simulation_overview.png`
  - `http://localhost:5000/TidyMesh/Sim/v2/visualization/qlearning_analysis.png`

#### Configuration Files
- **URL**: `http://localhost:5000/TidyMesh/Sim/v2/config/<filename>`
- **Method**: `GET`
- **Description**: Serves configuration files as JSON
- **Parameters**: 
  - `<filename>`: Name of the configuration file
- **Examples**:
  - `http://localhost:5000/TidyMesh/Sim/v2/config/roadZones.json`
  - `http://localhost:5000/TidyMesh/Sim/v2/config/trafficLights.json`
  - `http://localhost:5000/TidyMesh/Sim/v2/config/trashBinZones.json`

## Error Handling

### 404 Not Found
- **Description**: Returned when an endpoint doesn't exist
- **Response**: JSON with error message and list of available endpoints

### 500 Internal Server Error
- **Description**: Returned when an unexpected server error occurs
- **Response**: JSON with error message and timestamp

## Usage Examples

### Starting the Server
```bash
python tidymesh_api_server.py --host localhost --port 5000 --debug
```

### Example API Calls

#### Get API Information
```bash
curl http://localhost:5000/
```

#### Start a Simulation
```bash
curl -X POST http://localhost:5000/TidyMesh/Sim/v2/run \
  -H "Content-Type: application/json" \
  -d '{"n_trucks": 8, "n_bins": 50, "steps": 1500}'
```

#### Check Simulation Status
```bash
curl http://localhost:5000/TidyMesh/Sim/v2/run/status
```

#### Get Final State Data
```bash
curl http://localhost:5000/TidyMesh/Sim/v2/mas_final_state.json
```

#### Get Simulation History
```bash
curl http://localhost:5000/TidyMesh/Sim/v2/simulation_history.json
```

#### Get a Configuration File
```bash
curl http://localhost:5000/TidyMesh/Sim/v2/config/roadZones.json
```

## Notes

- All JSON responses include timestamps for data freshness tracking
- The API uses CORS enabling for Unity WebGL builds
- File caching is implemented for improved performance
- Simulation runs execute in separate threads to avoid blocking the API
- Parameter validation ensures safe ranges for all simulation parameters
- The server supports both development (debug mode) and production deployment
