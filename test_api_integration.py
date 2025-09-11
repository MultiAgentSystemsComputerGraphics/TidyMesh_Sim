#!/usr/bin/env python3
"""
Test script to demonstrate the API server and simulation integration.
This shows how both components work together.
"""

import requests
import json
import time

def test_api_integration():
    """Test the API server and simulation integration."""
    
    # Test 1: Check if API server is running
    print("=" * 60)
    print("TEST 1: Checking API server availability...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            api_info = response.json()
            print(f"   API Version: {api_info.get('version', 'unknown')}")
            print(f"   Available endpoints: {list(api_info.get('endpoints', {}).keys())}")
        else:
            print(f"❌ API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        print("   Start it with: python scripts/tidymesh_api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error checking API server: {e}")
        return False
    
    # Test 2: Send simulation parameters via POST
    print("\n" + "=" * 60)
    print("TEST 2: Sending simulation parameters via API...")
    
    # These are the parameters from your Postman example
    simulation_params = {
        "n_trucks": 15,
        "n_bins": 40,
        "n_tlights": 10,
        "n_obstacles": 10,
        "steps": 100000  # Longer simulation for more bins to be collected
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/TidyMesh/Sim/v3/run",
            json=simulation_params,
            timeout=10
        )
        
        if response.status_code == 202:  # Simulation started
            result = response.json()
            print("✅ Simulation started successfully")
            print(f"   Run ID: {result.get('run_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Parameters used:")
            for key, value in result.get('params', {}).items():
                print(f"     {key}: {value}")
            
            # Test 3: Monitor simulation status
            print("\n" + "=" * 60)
            print("TEST 3: Monitoring simulation progress...")
            
            max_wait = 60  # Maximum wait time in seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/run/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"   Simulation state: {status.get('state')}")
                    
                    if status.get('state') == 'completed':
                        print("✅ Simulation completed successfully")
                        break
                    elif status.get('state') == 'error':
                        print(f"❌ Simulation failed: {status.get('last_error')}")
                        break
                    elif status.get('state') == 'running':
                        print("   🔄 Simulation still running...")
                        time.sleep(5)  # Wait 5 seconds before checking again
                    else:
                        print(f"   Unknown state: {status.get('state')}")
                        break
                else:
                    print(f"❌ Error getting status: {status_response.status_code}")
                    break
            
            # Test 4: Check results
            print("\n" + "=" * 60)
            print("TEST 4: Checking simulation results...")
            
            try:
                final_state_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/mas_final_state.json")
                if final_state_response.status_code == 200:
                    final_state = final_state_response.json()
                    summary = final_state.get('data', {}).get('summary', {})
                    print("✅ Final state retrieved successfully")
                    print(f"   Total bins done: {summary.get('total_bins_done', 0)}")
                    print(f"   Total collected: {summary.get('total_collected', 0)}")
                    print(f"   Total distance: {summary.get('total_distance', 0)}")
                    print(f"   Simulation steps: {summary.get('steps', 0)}")
                else:
                    print(f"❌ Could not retrieve final state: {final_state_response.status_code}")
            except Exception as e:
                print(f"❌ Error retrieving results: {e}")
                
        else:
            print(f"❌ Failed to start simulation: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending simulation request: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ API integration test completed!")
    return True

if __name__ == "__main__":
    print("🚀 TidyMesh API Integration Test")
    print("Make sure the API server is running: python scripts/tidymesh_api_server.py")
    print("")
    
    success = test_api_integration()
    if success:
        print("\n🎉 All tests passed! The API integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the API server and try again.")