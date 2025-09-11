#!/usr/bin/env python3
"""
Quick test to send a POST request with specific parameters to verify they are used correctly.
"""

import requests
import json
import time

def test_postman_parameters():
    """Test sending the same parameters as shown in the Postman screenshot."""
    
    # The exact parameters from the Postman screenshot
    postman_params = {
        "n_trucks": 15,
        "n_bins": 40,
        "n_tlights": 10,
        "n_obstacles": 10,
        "steps": 100000
    }
    
    print("🧪 Testing API with Postman parameters...")
    print("Parameters being sent:")
    for key, value in postman_params.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        # Send POST request to API
        response = requests.post(
            "http://localhost:5000/TidyMesh/Sim/v3/run",
            json=postman_params,
            timeout=10
        )
        
        if response.status_code == 202:
            result = response.json()
            print("✅ API request successful!")
            print(f"Run ID: {result.get('run_id')}")
            print("Parameters confirmed by API:")
            api_params = result.get('params', {})
            for key, value in api_params.items():
                print(f"   {key}: {value}")
            
            # Check if parameters match what we sent
            match = True
            for key, sent_value in postman_params.items():
                api_value = api_params.get(key)
                if api_value != sent_value:
                    print(f"❌ MISMATCH: {key} - sent: {sent_value}, API got: {api_value}")
                    match = False
            
            if match:
                print("\n✅ All parameters match! API is receiving the correct values.")
            else:
                print("\n❌ Parameter mismatch detected!")
                
            # Wait a moment for simulation to start
            print("\n🔄 Waiting for simulation to start...")
            time.sleep(3)
            
            # Check status
            status_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/run/status")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"Simulation status: {status.get('state')}")
                if status.get('state') == 'running':
                    print("✅ Simulation is running with the sent parameters!")
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_postman_parameters()