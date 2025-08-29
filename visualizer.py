# visualizer.py
# Comprehensive visualization system for TidyMesh_Sim
# Includes real-time animation, post-processing analytics, and Q-learning graphs

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import os

class TidyMeshVisualizer:
    def __init__(self, json_file="results/simulation_data/mas_final_state.json", history_file="results/simulation_data/simulation_history.json"):
        self.json_file = json_file
        self.history_file = history_file
        # Ensure output directories exist
        os.makedirs("results/visualizations", exist_ok=True)
        self.load_data()
        
    def load_data(self):
        """Load simulation data from JSON files"""
        try:
            with open(self.json_file, 'r') as f:
                self.final_state = json.load(f)
            print(f"Loaded final state with {len(self.final_state['agents'])} agents")
        except FileNotFoundError:
            print(f"Warning: {self.json_file} not found. Run simulation first.")
            self.final_state = None
            
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
            print(f"Loaded history with {len(self.history['steps'])} time steps")
        except FileNotFoundError:
            print(f"Warning: {self.history_file} not found. Animation will not be available.")
            self.history = None

    def create_static_overview(self):
        """Create a static overview of the final simulation state"""
        if not self.final_state:
            print("No final state data available")
            return
            
        print("Creating static overview plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"TidyMesh Simulation Overview - Tick {self.final_state['tick']}", fontsize=16)
        
        try:
            # 1. Grid Layout with all agents
            print("  - Plotting grid layout...")
            self._plot_grid_layout(ax1)
            
            # 2. Truck Statistics
            print("  - Plotting truck statistics...")
            self._plot_truck_stats(ax2)
            
            # 3. Bin Status Distribution
            print("  - Plotting bin status...")
            self._plot_bin_status(ax3)
            
            # 4. Performance Metrics
            print("  - Plotting performance metrics...")
            self._plot_performance_metrics(ax4)
            
            plt.tight_layout()
            output_file = 'results/visualizations/simulation_overview.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"Static overview saved as {output_file}")
            
        except Exception as e:
            print(f"Error creating static overview: {e}")
            plt.close()  # Ensure figure is closed even on error

    def _plot_grid_layout(self, ax):
        """Plot the grid layout with all agents"""
        ax.set_title("Grid Layout - Final State")
        
        # Set grid boundaries
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Color scheme for different agent types
        colors = {
            'truck': 'blue',
            'trash_bin': 'green',
            'traffic_light': 'orange',
            'obstacle': 'red',
            'depot': 'purple',
            'dispatcher': 'black'
        }
        
        # Plot agents
        for agent in self.final_state['agents']:
            x, z = agent['x'], agent['z']
            agent_type = agent['type']
            
            if agent_type == 'truck':
                # Trucks as squares with load info
                size = 50 + agent.get('load', 0) * 20  # Size based on load
                ax.scatter(x, z, c=colors[agent_type], s=size, marker='s', alpha=0.8,
                          label=f"Truck (Load: {agent.get('load', 0):.1f})" if agent == self.final_state['agents'][self._get_first_truck_index()] else "")
                ax.text(x+0.3, z, f"T{agent['id'][-2:]}", fontsize=8)
                
            elif agent_type == 'trash_bin':
                # Bins with fill level color coding
                fill_level = agent.get('fill_level', 0)
                color_intensity = min(1.0, fill_level)
                ax.scatter(x, z, c=colors[agent_type], s=60, marker='o', 
                          alpha=0.3 + 0.7*color_intensity,
                          label="Trash Bin" if agent == self._get_first_bin() else "")
                
                # Status indicator
                if agent.get('ready_for_pickup', False):
                    ax.add_patch(Circle((x, z), 0.3, fill=False, edgecolor='red', linewidth=2))
                    
            elif agent_type == 'traffic_light':
                phase = agent.get('phase', 'G')
                tl_color = 'green' if phase == 'G' else 'red'
                ax.scatter(x, z, c=tl_color, s=40, marker='^', 
                          label="Traffic Light" if agent == self._get_first_traffic_light() else "")
                          
            else:
                ax.scatter(x, z, c=colors.get(agent_type, 'gray'), s=50, 
                          marker='o' if agent_type != 'depot' else 'D',
                          label=agent_type.replace('_', ' ').title() if self._is_first_of_type(agent) else "")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Z Coordinate")

    def _plot_truck_stats(self, ax):
        """Plot truck statistics"""
        trucks = [agent for agent in self.final_state['agents'] if agent['type'] == 'truck']
        
        truck_ids = [agent['id'] for agent in trucks]
        loads = [agent.get('load', 0) for agent in trucks]
        distances = [agent.get('total_distance', 0) for agent in trucks]
        collected = [agent.get('collected_bins', 0) for agent in trucks]
        
        x = np.arange(len(truck_ids))
        width = 0.25
        
        ax.bar(x - width, loads, width, label='Current Load', alpha=0.8)
        ax.bar(x, collected, width, label='Bins Collected', alpha=0.8)
        ax.bar(x + width, [d/50 for d in distances], width, label='Distance (÷50)', alpha=0.8)
        
        ax.set_xlabel('Trucks')
        ax.set_ylabel('Values')
        ax.set_title('Truck Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(truck_ids, rotation=45)
        ax.legend()

    def _plot_bin_status(self, ax):
        """Plot bin status distribution"""
        bins = [agent for agent in self.final_state['agents'] if agent['type'] == 'trash_bin']
        
        statuses = [agent.get('state', 'Unknown') for agent in bins]
        status_counts = pd.Series(statuses).value_counts()
        
        colors_status = {'Ready': 'orange', 'Done': 'green', 'Servicing': 'blue', 'Idle': 'gray'}
        pie_colors = [colors_status.get(status, 'gray') for status in status_counts.index]
        
        ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
               colors=pie_colors, startangle=90)
        ax.set_title('Bin Status Distribution')

    def _plot_performance_metrics(self, ax):
        """Plot overall performance metrics"""
        stats = self.final_state['simulation_stats']
        
        metrics = ['Total Bins Done', 'Total Collected', 'Total Distance', 'Open Tasks']
        values = [
            stats['total_bins_done'],
            stats['total_collected'], 
            stats['total_distance'],
            stats['open_tasks']
        ]
        
        bars = ax.bar(metrics, values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        ax.set_title('Overall Performance Metrics')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def create_animated_simulation(self, save_gif=True, interval=200):
        """Create animated visualization of the simulation"""
        if not self.history:
            print("No history data available for animation")
            return
            
        print("Creating animated simulation...")
        print(f"Processing {len(self.history['steps'])} time steps...")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            def animate(frame):
                if frame % 50 == 0:  # Progress indicator
                    print(f"  Animating frame {frame}/{len(self.history['steps'])}")
                
                ax.clear()
                
                # Get current step data
                step_data = self.history['steps'][frame]
                
                ax.set_xlim(-1, 21)
                ax.set_ylim(-1, 15)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f"TidyMesh Simulation - Tick {step_data['tick']}")
                
                # Plot agents for current frame
                for agent in step_data['agents']:
                    x, z = agent['x'], agent['z']
                    agent_type = agent['type']
                    
                    if agent_type == 'truck':
                        load = agent.get('load', 0)
                        size = 50 + load * 20
                        assigned = agent.get('assigned_bin', None)
                        color = 'darkblue' if assigned else 'blue'
                        ax.scatter(x, z, c=color, s=size, marker='s', alpha=0.8)
                        
                        # Show truck path if assigned
                        if assigned:
                            # Find bin position
                            for bin_agent in step_data['agents']:
                                if bin_agent['type'] == 'trash_bin' and bin_agent['id'] == assigned:
                                    ax.plot([x, bin_agent['x']], [z, bin_agent['z']], 
                                           'b--', alpha=0.5, linewidth=1)
                                    break
                                    
                    elif agent_type == 'trash_bin':
                        fill_level = agent.get('fill_level', 0)
                        ready = agent.get('ready_for_pickup', False)
                        color = 'red' if ready else 'green'
                        alpha = 0.3 + 0.7 * min(1.0, fill_level)
                        ax.scatter(x, z, c=color, s=60, marker='o', alpha=alpha)
                        
                    elif agent_type == 'depot':
                        ax.scatter(x, z, c='purple', s=100, marker='D')
                        
                    elif agent_type == 'obstacle':
                        ax.scatter(x, z, c='red', s=40, marker='x')
                        
                    elif agent_type == 'traffic_light':
                        phase = agent.get('phase', 'G')
                        color = 'green' if phase == 'G' else 'red'
                        ax.scatter(x, z, c=color, s=30, marker='^')
                    
                    elif agent_type == 'dispatcher':
                        ax.scatter(x, z, c='black', s=80, marker='*')
                
                # Add status text
                ready_bins = sum(1 for a in step_data['agents'] 
                               if a['type'] == 'trash_bin' and a.get('ready_for_pickup', False))
                done_bins = sum(1 for a in step_data['agents'] 
                              if a['type'] == 'trash_bin' and a.get('state') == 'Done')
                total_load = sum(a.get('load', 0) for a in step_data['agents'] 
                               if a['type'] == 'truck')
                
                ax.text(0.02, 0.98, f"Ready Bins: {ready_bins}\nDone Bins: {done_bins}\nTotal Load: {total_load:.1f}", 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            anim = animation.FuncAnimation(fig, animate, frames=len(self.history['steps']), 
                                         interval=interval, repeat=True)
            
            if save_gif:
                output_file = 'results/visualizations/simulation_animation.gif'
                print(f"Saving animation as {output_file}...")
                anim.save(output_file, writer='pillow', fps=5)
                print(f"Animation saved as {output_file}")
            
            plt.close()  # Close the figure to free memory
            return anim
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            plt.close()  # Ensure figure is closed even on error
            return None

    def create_qlearning_analysis(self):
        """Create Q-learning analysis plots"""
        if not self.final_state:
            print("No data available for Q-learning analysis")
            return
            
        print("Creating Q-learning analysis plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Q-Learning Analysis", fontsize=16)
        
        try:
            trucks = [agent for agent in self.final_state['agents'] if agent['type'] == 'truck']
            
            # 1. Action Distribution
            print("  - Plotting action distribution...")
            self._plot_action_distribution(ax1, trucks)
            
            # 2. Learning Progress (if available)
            print("  - Plotting learning progress...")
            self._plot_learning_progress(ax2, trucks)
            
            # 3. Efficiency Metrics
            print("  - Plotting efficiency metrics...")
            self._plot_efficiency_metrics(ax3, trucks)
            
            # 4. Spatial Analysis
            print("  - Plotting spatial analysis...")
            self._plot_spatial_analysis(ax4, trucks)
            
            plt.tight_layout()
            output_file = 'results/visualizations/qlearning_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"Q-learning analysis saved as {output_file}")
            
        except Exception as e:
            print(f"Error creating Q-learning analysis: {e}")
            plt.close()  # Ensure figure is closed even on error

    def _plot_action_distribution(self, ax, trucks):
        """Plot distribution of actions taken by trucks"""
        try:
            all_actions = []
            for truck in trucks:
                action_log = truck.get('action_log', [])
                actions = [entry.get('action', 'UNKNOWN') for entry in action_log if isinstance(entry, dict)]
                all_actions.extend(actions)
            
            if all_actions:
                action_counts = pd.Series(all_actions).value_counts()
                bars = ax.bar(action_counts.index, action_counts.values, alpha=0.7)
                ax.set_title('Action Distribution')
                ax.set_xlabel('Actions')
                ax.set_ylabel('Frequency')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, action_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{count}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No action data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Action Distribution - No Data')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Action Distribution - Error')

    def _plot_learning_progress(self, ax, trucks):
        """Plot learning progress over time"""
        # This would require storing Q-values over time
        # For now, show distance efficiency over time
        
        for i, truck in enumerate(trucks):
            action_log = truck.get('action_log', [])
            if action_log:
                ticks = [entry['tick'] for entry in action_log]
                loads = [entry['load'] for entry in action_log]
                ax.plot(ticks, loads, label=f"Truck {truck['id'][-2:]}", alpha=0.7)
        
        ax.set_title('Load Over Time (Learning Progress)')
        ax.set_xlabel('Simulation Tick')
        ax.set_ylabel('Truck Load')
        ax.legend()

    def _plot_efficiency_metrics(self, ax, trucks):
        """Plot efficiency metrics"""
        truck_ids = [truck['id'][-2:] for truck in trucks]
        distances = [truck.get('total_distance', 0) for truck in trucks]
        collected = [truck.get('collected_bins', 0) for truck in trucks]
        
        efficiency = [c/max(1, d/10) for c, d in zip(collected, distances)]  # bins per 10 distance units
        
        bars = ax.bar(truck_ids, efficiency, alpha=0.7, color='green')
        ax.set_title('Truck Efficiency (Bins/Distance)')
        ax.set_xlabel('Truck ID')
        ax.set_ylabel('Efficiency Score')
        
        # Add value labels
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{eff:.2f}', ha='center', va='bottom')

    def _plot_spatial_analysis(self, ax, trucks):
        """Plot spatial movement analysis"""
        ax.set_title('Truck Movement Heatmap')
        
        # Create heatmap of truck positions
        x_positions = []
        y_positions = []
        
        for truck in trucks:
            action_log = truck.get('action_log', [])
            for entry in action_log:
                pos = entry.get('pos', [0, 0])
                x_positions.append(pos[0])
                y_positions.append(pos[1])
        
        if x_positions and y_positions:
            ax.hist2d(x_positions, y_positions, bins=20, alpha=0.7, cmap='Blues')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
        else:
            ax.text(0.5, 0.5, 'No position data available', ha='center', va='center', transform=ax.transAxes)

    # Helper methods
    def _get_first_truck_index(self):
        for i, agent in enumerate(self.final_state['agents']):
            if agent['type'] == 'truck':
                return i
        return 0
    
    def _get_first_bin(self):
        for agent in self.final_state['agents']:
            if agent['type'] == 'trash_bin':
                return agent
        return None
    
    def _get_first_traffic_light(self):
        for agent in self.final_state['agents']:
            if agent['type'] == 'traffic_light':
                return agent
        return None
    
    def _is_first_of_type(self, target_agent):
        for agent in self.final_state['agents']:
            if agent['type'] == target_agent['type']:
                return agent['id'] == target_agent['id']
        return False

    def create_all_visualizations(self):
        """Create all available visualizations"""
        print("Creating all available visualizations...")
        print("=" * 50)
        
        print("1. Creating static overview...")
        self.create_static_overview()
        
        print("\n2. Creating Q-learning analysis...")
        self.create_qlearning_analysis()
        
        if self.history:
            print("\n3. Creating animated simulation...")
            self.create_animated_simulation()
        else:
            print("\n3. Skipping animation (no history data)")
        
        print("\n" + "=" * 50)
        print("All visualizations complete!")
        print("Generated files:")
        if os.path.exists('results/visualizations/simulation_overview.png'):
            print("✓ results/visualizations/simulation_overview.png")
        if os.path.exists('results/visualizations/qlearning_analysis.png'):
            print("✓ results/visualizations/qlearning_analysis.png")
        if os.path.exists('results/visualizations/simulation_animation.gif'):
            print("✓ results/visualizations/simulation_animation.gif")
        print("\nOpen these files to view your simulation results!")

# Standalone usage
if __name__ == "__main__":
    visualizer = TidyMeshVisualizer()
    visualizer.create_all_visualizations()
