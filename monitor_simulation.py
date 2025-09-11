#!/usr/bin/env python3
"""
Monitor the current API simulation and show the results.
"""

import requests
import time
import json

def monitor_api_simulation():
    """Monitor the running API simulation."""
    
    print("🔍 Monitoring current API simulation...")
    
    # Get current status
    status_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/run/status")
    if status_response.status_code == 200:
        status = status_response.json()
        
        print(f"Run ID: {status.get('run_id')}")
        print(f"State: {status.get('state')}")
        print(f"Started: {status.get('started_at')}")
        
        # Show parameters that are actually running
        params = status.get('params', {})
        print("\n📊 Current simulation parameters:")
        print(f"   Trucks: {params.get('n_trucks', 'unknown')}")
        print(f"   Bins: {params.get('n_bins', 'unknown')}")
        print(f"   Traffic Lights: {params.get('n_tlights', 'unknown')}")
        print(f"   Obstacles: {params.get('n_obstacles', 'unknown')}")
        print(f"   Steps: {params.get('steps', 'unknown')}")
        
        # Wait for completion if running
        if status.get('state') == 'running':
            print("\n⏳ Waiting for simulation to complete...")
            
            while True:
                time.sleep(5)
                status_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/run/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    state = status.get('state')
                    print(f"   Status: {state}")
                    
                    if state == 'completed':
                        print("✅ Simulation completed!")
                        
                        # Get results
                        final_state_response = requests.get("http://localhost:5000/TidyMesh/Sim/v3/mas_final_state.json")
                        if final_state_response.status_code == 200:
                            final_data = final_state_response.json()
                            stats = final_data.get('data', {}).get('simulation_stats', {})
                            
                            print("\n📈 Final Results:")
                            print(f"   Total bins done: {stats.get('total_bins_done', 0)}")
                            print(f"   Total collected: {stats.get('total_collected', 0)}")
                            print(f"   Total distance: {stats.get('total_distance', 0)}")
                            print(f"   Steps completed: {final_data.get('data', {}).get('tick', 0)}")
                            
                        break
                    elif state == 'error':
                        print(f"❌ Simulation failed: {status.get('last_error')}")
                        break
                else:
                    print("❌ Error getting status")
                    break
        elif status.get('state') == 'completed':
            print("✅ Simulation already completed!")
        else:
            print(f"ℹ  Simulation state: {status.get('state')}")
    else:
        print("❌ Could not get simulation status")

if __name__ == "__main__":
    monitor_api_simulation()