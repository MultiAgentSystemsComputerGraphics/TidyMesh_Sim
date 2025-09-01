#   /$$$$$$$$ /$$       /$$           /$$      /$$                     /$$      
#  |__  $$__/|__/      | $$          | $$$    /$$$                    | $$      
#     | $$    /$$  /$$$$$$$ /$$   /$$| $$$$  /$$$$  /$$$$$$   /$$$$$$$| $$$$$$$ 
#     | $$   | $$ /$$__  $$| $$  | $$| $$ $$/$$ $$ /$$__  $$ /$$_____/| $$__  $$
#     | $$   | $$| $$  | $$| $$  | $$| $$  $$$| $$| $$$$$$$$|  $$$$$$ | $$  \ $$
#     | $$   | $$| $$  | $$| $$  | $$| $$\  $ | $$| $$_____/ \____  $$| $$  | $$
#     | $$   | $$|  $$$$$$$|  $$$$$$$| $$ \/  | $$|  $$$$$$$ /$$$$$$$/| $$  | $$
#     |__/   |__/ \_______/ \____  $$|__/     |__/ \_______/|_______/ |__/  |__/
#                           /$$  | $$                                           
#                          |  $$$$$$/                                           
#                           \______/                                            

# DEBUGGING SCRIPT
# For NDS Cognitive Labs Mexico

# By: 
# Santiago Quintana Moreno      A01571222
# Sergio Rodríguez Pérez        A00838856
# Rodrigo González de la Garza  A00838952
# Diego Gaitan Sanchez          A01285960
# Miguel Ángel Álvarez Hermida  A01722925

# COPYRIGHT 2025 TIDYMESH INC. ALL RIGHTS RESERVED. 2025 


import json
from collections import Counter

def analyze_history():
    """Analyze the history data to debug animation issues"""
    print("Analyzing simulation history...")
    
    try:
        with open('simulation_history.json', 'r') as f:
            history = json.load(f)
        
        steps = history['steps']
        print(f"Total steps: {len(steps)}")
        
        if not steps:
            print("ERROR: No steps in history!")
            return
        
        # Analyze first step
        first_step = steps[0]
        print(f"\nFirst step (tick {first_step['tick']}):")
        print(f"  Agents: {len(first_step['agents'])}")
        
        # Count agent types
        agent_types = Counter(agent['type'] for agent in first_step['agents'])
        print(f"  Agent types: {dict(agent_types)}")
        
        # Show sample agents
        print("\nSample agents from first step:")
        for i, agent in enumerate(first_step['agents'][:10]):
            print(f"  {i+1}. ID: {agent['id']}, Type: {agent['type']}, Pos: ({agent['x']}, {agent['z']})")
        
        # Check if agents move
        if len(steps) > 1:
            last_step = steps[-1]
            print(f"\nLast step (tick {last_step['tick']}):")
            
            # Check for position changes
            first_positions = {agent['id']: (agent['x'], agent['z']) for agent in first_step['agents']}
            last_positions = {agent['id']: (agent['x'], agent['z']) for agent in last_step['agents']}
            
            moved_agents = 0
            for agent_id in first_positions:
                if agent_id in last_positions and first_positions[agent_id] != last_positions[agent_id]:
                    moved_agents += 1
            
            print(f"  Agents that moved: {moved_agents}/{len(first_positions)}")
            
            # Show some movement examples
            print("\nMovement examples:")
            count = 0
            for agent_id in first_positions:
                if count >= 5:
                    break
                if agent_id in last_positions and first_positions[agent_id] != last_positions[agent_id]:
                    first_pos = first_positions[agent_id]
                    last_pos = last_positions[agent_id]
                    print(f"  {agent_id}: {first_pos} -> {last_pos}")
                    count += 1
        
        print("\nHistory analysis complete!")
        
    except Exception as e:
        print(f"Error analyzing history: {e}")

def test_animation_frame():
    """Test a single animation frame"""
    print("\nTesting animation frame...")
    
    try:
        from visualizer import TidyMeshVisualizer
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        visualizer = TidyMeshVisualizer()
        if not visualizer.history:
            print("No history data available")
            return
        
        # Create a test plot for one frame
        fig, ax = plt.subplots(figsize=(10, 6))
        
        step_data = visualizer.history['steps'][0]
        print(f"Testing frame with {len(step_data['agents'])} agents")
        
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Test Frame - Tick {step_data['tick']}")
        
        agent_count = Counter()
        
        for agent in step_data['agents']:
            x, z = agent['x'], agent['z']
            agent_type = agent['type']
            agent_count[agent_type] += 1
            
            if agent_type == 'truck':
                load = agent.get('load', 0)
                size = 50 + load * 20
                ax.scatter(x, z, c='blue', s=size, marker='s', alpha=0.8, label='Truck' if agent_count[agent_type] == 1 else "")
                
            elif agent_type == 'trash_bin':
                fill_level = agent.get('fill_level', 0)
                ready = agent.get('ready_for_pickup', False)
                color = 'red' if ready else 'green'
                alpha = 0.3 + 0.7 * min(1.0, fill_level)
                ax.scatter(x, z, c=color, s=60, marker='o', alpha=alpha, label='Bin' if agent_count[agent_type] == 1 else "")
                
            elif agent_type == 'depot':
                ax.scatter(x, z, c='purple', s=100, marker='D', label='Depot' if agent_count[agent_type] == 1 else "")
                
            elif agent_type == 'obstacle':
                ax.scatter(x, z, c='red', s=40, marker='x', label='Obstacle' if agent_count[agent_type] == 1 else "")
                
            elif agent_type == 'traffic_light':
                phase = agent.get('phase', 'G')
                color = 'green' if phase == 'G' else 'red'
                ax.scatter(x, z, c=color, s=30, marker='^', label='Traffic Light' if agent_count[agent_type] == 1 else "")
                
            elif agent_type == 'dispatcher':
                ax.scatter(x, z, c='black', s=80, marker='*', label='Dispatcher' if agent_count[agent_type] == 1 else "")
        
        ax.legend()
        plt.savefig('test_frame.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Test frame saved as test_frame.png")
        print(f"Agent counts: {dict(agent_count)}")
        
    except Exception as e:
        print(f"Error testing animation frame: {e}")

if __name__ == "__main__":
    analyze_history()
    test_animation_frame()
