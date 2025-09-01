# test_api_client.py
# Test client for TidyMesh API Server
# Demonstrates how Unity can interact with the simulation API
# Author: TidyMesh Development Team

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"
ENDPOINTS = {
    "info": f"{API_BASE_URL}/",
    "health": f"{API_BASE_URL}/health", 
    "status": f"{API_BASE_URL}/status",
    "final_state": f"{API_BASE_URL}/TidyMesh/Sim/v2/mas_final_state.json",
    "simulation_history": f"{API_BASE_URL}/TidyMesh/Sim/v2/simulation_history.json"
}

def test_endpoint(name, url, method="GET", data=None):
    """Test a single API endpoint"""
    print(f"\n🔍 Testing {name} endpoint...")
    print(f"   URL: {url}")
    print(f"   Method: {method}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print(f"   ✅ Success - Response size: {len(str(json_data))} characters")
                
                # Show key information based on endpoint
                if name == "API Info":
                    print(f"   📋 API Version: {json_data.get('version', 'N/A')}")
                    print(f"   📝 Description: {json_data.get('description', 'N/A')}")
                elif name == "Health Check":
                    files = json_data.get('files', {})
                    print(f"   📁 Final state exists: {files.get('final_state_exists', False)}")
                    print(f"   📁 History exists: {files.get('simulation_history_exists', False)}")
                elif name == "Final State":
                    data_section = json_data.get('data', {})
                    if data_section:
                        print(f"   📊 Agents: {len(data_section.get('agents', []))}")
                        print(f"   🗓️  Timestamp: {json_data.get('timestamp', 'N/A')}")
                elif name == "Simulation History":
                    data_section = json_data.get('data', {})
                    if data_section:
                        steps = data_section.get('steps', [])
                        print(f"   📈 Steps recorded: {len(steps)}")
                        print(f"   🗓️  Timestamp: {json_data.get('timestamp', 'N/A')}")
                
            except json.JSONDecodeError:
                print(f"   ⚠️  Non-JSON response: {response.text[:100]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   📄 Error message: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   📄 Raw response: {response.text[:200]}...")
                
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Cannot connect to server")
        print(f"   💡 Hint: Make sure the API server is running with: python tidymesh_api_server.py")
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout Error: Server took too long to respond")
    except Exception as e:
        print(f"   ❌ Unexpected Error: {str(e)}")

def test_post_example():
    """Test POST functionality with example data"""
    print(f"\n📤 Testing POST functionality...")
    
    # Example data that a simulation might send
    example_final_state = {
        "simulation_id": "test_run_001",
        "timestamp": "2025-08-31T20:30:00",
        "total_steps": 1200,
        "agents": [
            {
                "id": "TRUCK_50",
                "type": "GarbageTruck",
                "position": [250, 200],
                "load": 2.5,
                "bins_completed": 3,
                "total_distance": 450.7
            },
            {
                "id": "BIN_001", 
                "type": "TrashBin",
                "position": [100, 150],
                "fill_level": 0.0,
                "status": "done"
            }
        ],
        "performance": {
            "bins_completed": 6,
            "total_distance": 2485.3,
            "simulation_time": 60.5
        }
    }
    
    test_endpoint("Final State POST", ENDPOINTS["final_state"], "POST", example_final_state)

def main():
    """Main test function"""
    print("🚀 TidyMesh API Client Test")
    print("=" * 50)
    print("This script tests the TidyMesh API endpoints")
    print("to demonstrate Unity integration capabilities.")
    print("=" * 50)
    
    # Test all GET endpoints
    test_endpoint("API Info", ENDPOINTS["info"])
    test_endpoint("Health Check", ENDPOINTS["health"])
    test_endpoint("Status", ENDPOINTS["status"])
    test_endpoint("Final State", ENDPOINTS["final_state"])
    test_endpoint("Simulation History", ENDPOINTS["simulation_history"])
    
    # Test POST functionality
    test_post_example()
    
    print(f"\n" + "=" * 50)
    print("🎮 Unity Integration Guide:")
    print("=" * 50)
    print("1. Start the API server: python tidymesh_api_server.py")
    print("2. In Unity, use UnityWebRequest to GET simulation data:")
    print(f"   - Final State: {ENDPOINTS['final_state']}")
    print(f"   - History: {ENDPOINTS['simulation_history']}")
    print("3. Use POST requests to send new simulation data")
    print("4. Parse the JSON response to visualize the multi-agent system")
    print(f"\n✨ Enhanced Multi-Layered Q-Learning data available!")

if __name__ == "__main__":
    main()
