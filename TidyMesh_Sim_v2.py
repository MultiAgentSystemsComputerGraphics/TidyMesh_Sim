#   /$$$$$$$$ /$$       /$$           /$$      /$$                     /$$      
#  |__  $$__/|__/      | $$          | $$$    /$$$                    | $$      
#     | $$    /$$  /$$$$$$$ /$$   /$$| $$$$  /$$$$  /$$$$$$   /$$$$$$$| $$$$$$$ 
#     | $$   | $$ /$$__  $$| $$  | $$| $$ $$/$$ $$ /$$__  $$ /$$_____/| $$__  $$
#     | $$   | $$| $$  | $$| $$  | $$| $$  $$$| $$| $$$$$$$$|  $$$$$$ | $$  \ $$
#     | $$   | $$| $$  | $$| $$  | $$| $$\  $ | $$| $$_____/ \____  $$| $$  | $$
#     | $$   | $$|  $$$$$$$|  $$$$$$$| $$ \/  | $$|  $$$$$$$ /$$$$$$$/| $$  | $$    v2
#     |__/   |__/ \_______/ \____  $$|__/     |__/ \_______/|_______/ |__/  |__/
#                           /$$  | $$                                           
#                          |  $$$$$$/                                           
#                           \______/                                            

# TIDYMESH SIMULATION - Garbage collection system
# For NDS Cognitive Labs Mexico

# By: 
# Santiago Quintana Moreno      A01571222
# Sergio Rodríguez Pérez        A00838856
# Rodrigo González de la Garza  A00838952
# Diego Gaitan Sanchez          A01285960
# Miguel Ángel Álvarez Hermida  A01722925

# COPYRIGHT 2025 TIDYMESH INC. ALL RIGHTS RESERVED. 2025 

import agentpy as ap
import numpy as np
import json
import math
import random
import time
from collections import defaultdict, deque

# -------------------------
# Utility helpers
# -------------------------

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def transform_coordinates(json_x, json_z, offset_x, offset_z):
    """Transform JSON coordinates to grid coordinates"""
    grid_x = int(json_x + offset_x)
    grid_z = int(json_z + offset_z)
    return grid_x, grid_z

def is_valid_grid_position(x, z, width, height):
    """Check if grid coordinates are within bounds"""
    return 0 <= x < width and 0 <= z < height

# -------------------------
# Agent base with static IDs
# -------------------------

class BaseEntity(ap.Agent):
    def setup(self):
        # Each entity carries a static id that stays fixed for the run.
        # The model assigns this before placement.
        assert hasattr(self, "static_id"), "static_id must be set by the model"
        self.entity_type = getattr(self, "entity_type", "entity")
        self.label = getattr(self, "label", self.entity_type)

    @property
    def pos(self):
        # AgentPy grid stores positions in model.grid.positions
        return tuple(self.model.grid.positions[self])

    def pos_xz(self):
        x, y = self.pos
        return int(x), int(y)  # Map y -> z in Unity terms

# -------------------------
# Depot
# -------------------------

class Depot(BaseEntity):
    def setup(self):
        super().setup()
        self.entity_type = "depot"

# -------------------------
# Traffic Light (periodic R/G)
# -------------------------

class TrafficLight(BaseEntity):
    def setup(self):
        super().setup()
        self.entity_type = "traffic_light"
        self.cycle_len = self.p.tl_cycle  # total ticks per cycle
        self.green_len = self.p.tl_green  # green ticks
        self.phase = "G"  # start green
        self.t = 0

    def step(self):
        self.t = (self.t + 1) % self.cycle_len
        self.phase = "G" if self.t < self.green_len else "R"

# -------------------------
# Dynamic Obstacle (random walk)
# -------------------------

class DynamicObstacle(BaseEntity):
    def setup(self):
        super().setup()
        self.entity_type = "obstacle"

    def step(self):
        # Random move with some probability
        if random.random() < self.p.obstacle_move_prob:
            x, y = self.pos
            choices = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            random.shuffle(choices)
            for nx, ny in choices:
                if self.model.in_bounds((nx, ny)) and self.model.cell_free((nx, ny)):
                    self.model.grid.move_to(self, (nx, ny))
                    break

# -------------------------
# Trash Bin (task source)
# -------------------------

class TrashBin(BaseEntity):
    def setup(self):
        super().setup()
        self.entity_type = "trash_bin"
        # Simple fullness dynamics
        self.fill = random.uniform(self.p.bin_init_min, self.p.bin_init_max)
        self.threshold = self.p.bin_threshold
        # Set initial state based on fill level
        if self.fill >= self.threshold:
            self.state = "Ready"
        else:
            self.state = "Idle"  # Idle -> Ready -> Servicing -> Done

    def step(self):
        if self.state == "Idle":
            # bins fill over time
            self.fill = clamp(self.fill + self.p.bin_fill_rate, 0, 1.5)
            if self.fill >= self.threshold:
                self.state = "Ready"
                # Notify dispatcher (READY)
                self.model.dispatcher_queue.append(("READY", self, {"vol": self.fill}))
        elif self.state in ("Servicing", "Done"):
            pass
    
    def notify_ready(self):
        """Called after setup to notify dispatcher of initially ready bins"""
        if self.state == "Ready":
            self.model.dispatcher_queue.append(("READY", self, {"vol": self.fill}))

# -------------------------
# Dispatcher (Contract Net coordinator)
# -------------------------

class Dispatcher(BaseEntity):
    def setup(self):
        super().setup()
        self.entity_type = "dispatcher"
        self.open_tasks = set()       # bins awaiting assignment
        self.pending_cfp = dict()     # task -> dict(bids={truck:cost}, tick)
        self.awards = dict()          # task -> truck
        self.fairness_ledger = defaultdict(int)  # count assigned per truck
        self.cfp_timeout = self.p.cfp_timeout

    def step(self):
        # Process inbound messages from bins & trucks
        messages_to_process = list(self.model.dispatcher_queue)
        self.model.dispatcher_queue.clear()
        
        for msg, sender, payload in messages_to_process:
            if msg == "READY":  # bin ready
                self.open_tasks.add(sender)
            elif msg == "INFORM-DONE":
                b = payload["bin"]
                if b in self.open_tasks:
                    self.open_tasks.discard(b)
                if b in self.awards:
                    self.awards.pop(b, None)
            elif msg == "INFORM-FAIL":
                b = payload["bin"]
                # Re-issue by re-opening task (dead-letter queue could be used)
                self.open_tasks.add(b)
                self.awards.pop(b, None)
            elif msg == "PROPOSE":
                # Handle proposals for pending CFPs
                b = payload["bin"]
                if b in self.pending_cfp:
                    self.pending_cfp[b]["bids"][sender] = payload["cost"]

        # Issue CFPs for newly open & un-auctioned tasks
        for b in list(self.open_tasks):
            if b not in self.pending_cfp and b not in self.awards:
                self.pending_cfp[b] = {"bids": {}, "tick": self.model.t}
                # Broadcast CFP: trucks will add PROPOSE into dispatcher_queue
                for truck in self.model.trucks:
                    truck.receive_cfp(b)

        # Handle CFP timeouts & awards
        to_close = []
        for b, info in self.pending_cfp.items():
            # Decide if ready to award
            if (self.model.t - info["tick"]) >= self.cfp_timeout or len(info["bids"]) == len(self.model.trucks):
                if info["bids"]:
                    # Filter out trucks that are already assigned
                    available_bids = {tr: cost for tr, cost in info["bids"].items() 
                                    if tr.assigned_bin is None}
                    
                    if available_bids:
                        # Select min-cost proposal, with fairness penalty
                        fairness_penalty = {tr: self.fairness_ledger[tr]*self.p.fairness_alpha for tr in available_bids}
                        scored = [(c + fairness_penalty[tr], tr) for tr, c in available_bids.items()]
                        scored.sort(key=lambda x: (x[0], self.model.truck_load(x[1])))
                        best_score, best_truck = scored[0]
                        self.awards[b] = best_truck
                        self.fairness_ledger[best_truck] += 1
                        best_truck.receive_award(b)
                        # Reject others (no explicit message needed here)
                        to_close.append(b)
                    else:
                        # No available trucks -> leave open to retry later
                        to_close.append(b)
                else:
                    # No bids -> leave open to retry later
                    to_close.append(b)

        for b in to_close:
            self.pending_cfp.pop(b, None)

# -------------------------
# Garbage Truck (Q-Learning)
# -------------------------

class GarbageTruck(BaseEntity):
    ACTIONS = ("UP","DOWN","LEFT","RIGHT","WAIT","PICK","DROP","CHARGE","EXPLORE","RETREAT")

    def setup(self):
        super().setup()
        self.entity_type = "truck"
        # Capabilities
        self.capacity = self.p.truck_capacity
        self.load = 0.0
        self.speed = self.p.truck_speed  # cells per step

        # Task, motion & energy
        self.assigned_bin = None
        self.state = "Idle"
        self.total_distance = 0
        self.last_action = "WAIT"  # Track last action taken
        self.action_log = []  # Track all actions for debugging
        self.energy = self.p.truck_energy_max
        self.active = True  # Disabled after crashes

        # Enhanced Q-learning params with decay
        self.alpha = self.p.q_alpha
        self.alpha_decay = 0.9995  # Slowly reduce learning rate
        self.min_alpha = 0.01
        self.gamma = self.p.q_gamma
        self.epsilon = self.p.q_epsilon
        self.epsilon_decay = 0.999  # Reduce exploration over time
        self.min_epsilon = 0.01
        
        # Multi-layered Q-tables for different contexts
        self.q_navigation = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})
        self.q_exploration = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})
        self.q_emergency = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})
        
        # Enhanced state tracking
        self.corner_timer = 0  # Time spent in corners
        self.position_history = deque(maxlen=20)  # Track recent positions
        self.stuck_counter = 0  # Track if agent is stuck
        self.last_positions = deque(maxlen=5)  # For detecting loops
        self.exploration_map = defaultdict(int)  # Track visited cells
        self.danger_zones = set()  # Cells to avoid
        
        # Performance metrics
        self.efficiency_score = 0.0
        self.response_time = 0
        self.last_reward = 0
        
        # Stats
        self.collected_bins = 0

    # ------------- Contract Net participation -------------

    def receive_cfp(self, bin_agent):
        # Only bid if not already assigned to a bin
        if self.assigned_bin is not None:
            return  # Already busy with another bin
        
        # Decide cost if feasible, otherwise REFUSE
        if self.load >= self.capacity * 0.95:
            return  # insufficient capacity
        cost = self.estimate_cost_to_bin(bin_agent)
        self.model.dispatcher_queue.append(("PROPOSE", self, {"bin": bin_agent, "cost": cost}))

    def receive_award(self, bin_agent):
        # Only accept award if not already assigned
        if self.assigned_bin is not None:
            return
            
        self.assigned_bin = bin_agent
        if self.state in ("Idle","Patrolling","Bidding"):
            self.state = "Navigating"

    def estimate_cost_to_bin(self, bin_agent):
        # Simple ETA + detour + load penalty (per your spec)
        my_pos = self.pos
        b_pos = bin_agent.pos
        eta = manhattan(my_pos, b_pos)
        detour = 0  # grid-shortest assumption
        load_pen = (self.load / max(1e-6, self.capacity)) * 2
        return eta + detour + load_pen

    # ---------------- Enhanced Q-learning core ---------------------
    
    def is_corner(self, pos=None):
        """Check if position is a corner (not just quadrant)"""
        if pos is None:
            pos = self.pos
        x, y = pos
        
        # Use actual simulation boundaries for corner detection
        margin = 15  # Distance from edge to be considered a corner
        
        corners = [
            # Top-left corner
            (x <= margin and y >= self.model.p.height - margin),
            # Top-right corner  
            (x >= self.model.p.width - margin and y >= self.model.p.height - margin),
            # Bottom-left corner
            (x <= margin and y <= margin),
            # Bottom-right corner
            (x >= self.model.p.width - margin and y <= margin)
        ]
        
        return any(corners)
    
    def get_local_density(self, radius=3):
        """Calculate local density of agents around current position"""
        x, y = self.pos
        density = 0
        total_cells = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self.model.in_bounds((nx, ny)):
                    total_cells += 1
                    if not self.model.cell_free((nx, ny)):
                        density += 1
        
        return density / max(total_cells, 1)
    
    def get_traffic_pressure(self):
        """Analyze nearby traffic lights and movement options"""
        x, y = self.pos
        red_lights_nearby = 0
        blocked_directions = 0
        
        # Check immediate area for red lights
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if self.model.in_bounds((nx, ny)):
                for tl in self.model.traffic_lights:
                    if tl.pos == (nx, ny) and tl.phase == "R":
                        red_lights_nearby += 1
                        blocked_directions += 1
                    elif not self.model.cell_free((nx, ny)):
                        blocked_directions += 1
        
        return red_lights_nearby, blocked_directions / 4.0
    
    def perceive_enhanced_state(self):
        """Enhanced state perception with much more granular information"""
        x, y = self.pos
        
        # Basic position info
        grid_x = min(9, x // (self.model.p.width // 10))
        grid_y = min(9, y // (self.model.p.height // 10))
        
        # Target analysis
        target = None
        target_distance = 0
        target_direction = "NONE"
        
        if self.energy <= self.p.energy_threshold or self.state in ("NeedCharge", "Charging"):
            target = self.model.dispatcher.pos
        elif self.load >= self.capacity * self.p.unload_threshold:
            target = self.model.depot.pos
        elif self.assigned_bin and self.assigned_bin.state in ("Ready","Servicing"):
            target = self.assigned_bin.pos
        
        if target:
            target_distance = min(20, manhattan(self.pos, target))  # Capped for state space
            dx, dy = target[0] - x, target[1] - y
            if abs(dx) > abs(dy):
                target_direction = "E" if dx > 0 else "W"
            elif abs(dy) > 0:
                target_direction = "N" if dy > 0 else "S"
            else:
                target_direction = "HERE"
        
        # Environmental pressures
        red_lights, blocked_ratio = self.get_traffic_pressure()
        local_density = min(10, int(self.get_local_density() * 10))  # 0-10 scale
        
        # State flags
        has_load = min(3, int(self.load / self.capacity * 3))  # 0-3 scale
        assigned = 1 if self.assigned_bin else 0
        corner_state = 1 if self.is_corner() else 0
        
        # Movement analysis
        stuck_state = 1 if self.stuck_counter > 3 else 0
        repeated_position = 1 if len(self.last_positions) >= 3 and len(set(self.last_positions)) <= 2 else 0
        
        return (grid_x, grid_y, target_direction, target_distance, 
                red_lights, local_density, has_load, assigned, 
                corner_state, stuck_state, repeated_position, self.last_action)
    
    def select_q_table(self, state):
        """Select appropriate Q-table based on current context"""
        _, _, _, target_distance, red_lights, local_density, _, _, corner_state, stuck_state, repeated_position, _ = state
        
        # Emergency situations: corners, stuck, high pressure
        if corner_state or stuck_state or repeated_position or (red_lights >= 2 and local_density >= 7):
            return self.q_emergency
        
        # Navigation when we have a clear target
        if target_distance > 0 and target_distance <= 15:
            return self.q_navigation
        
        # Exploration for general movement
        return self.q_exploration
    
    def choose_action_enhanced(self, state):
        """Enhanced action selection with context awareness"""
        q_table = self.select_q_table(state)
        
        # Extract state components
        grid_x, grid_y, target_direction, target_distance, red_lights, local_density, has_load, assigned, corner_state, stuck_state, repeated_position, last_action = state
        
        # DEBUG: Track if we're in a corner
        if self.is_corner():
            self.corner_timer += 1
        else:
            self.corner_timer = 0
            
        # PRIORITY 1: Cliff avoidance - force immediate escape from corners
        if self.is_corner():
            # Force immediate escape from corners - no tolerance for staying
            x, y = self.pos
            escape_moves = []
            margin = 15  # Same margin as in is_corner
            
            # Check all four directions for escape
            if x <= margin and self._can_move("RIGHT"):  # Left side corners
                escape_moves.append("RIGHT")
            if x >= self.model.p.width - margin and self._can_move("LEFT"):  # Right side corners
                escape_moves.append("LEFT")
            if y <= margin and self._can_move("UP"):  # Bottom corners
                escape_moves.append("UP")
            if y >= self.model.p.height - margin and self._can_move("DOWN"):  # Top corners
                escape_moves.append("DOWN")
            
            # If we have escape options, use them immediately
            if escape_moves:
                return random.choice(escape_moves)
            
            # If no direct escape, try any movement
            all_moves = ["UP", "LEFT", "RIGHT"]
            valid_moves = [move for move in all_moves if self._can_move(move)]
            if valid_moves:
                return random.choice(valid_moves)
        
        # PRIORITY 2: Handle stuck situations with forced movement
        if self.stuck_counter > 5 or (len(self.last_positions) >= 3 and len(set(self.last_positions)) <= 1):
            # Force movement when stuck
            movement_actions = ["UP", "LEFT", "RIGHT"]
            valid_moves = [action for action in movement_actions if self._can_move(action)]
            
            if valid_moves:
                # Reset stuck counter on successful movement attempt
                self.stuck_counter = max(0, self.stuck_counter - 1)
                return random.choice(valid_moves)
        
        # PRIORITY 3: Direct task execution when at target
        if target_direction == "HERE":
            if self.energy <= self.p.energy_threshold or self.state in ("NeedCharge", "Charging"):
                return "WAIT"
            if has_load > 0:
                # At depot, should drop
                return "DROP"
            elif assigned:
                # At assigned bin, should pick
                return "PICK"
        
        # PRIORITY 4: Direct navigation to target when we have one
        if target_direction != "NONE" and target_distance > 0:
            action = self._get_direct_action(target_direction)
            if action and self._can_move(action):
                return action
            else:
                # Try alternative directions if direct path is blocked
                alt_actions = ["UP", "LEFT", "RIGHT"]
                alt_actions.remove(action) if action in alt_actions else None
                valid_alts = [a for a in alt_actions if self._can_move(a)]
                if valid_alts:
                    return random.choice(valid_alts)
        
        # PRIORITY 5: Use Q-learning for general exploration
        if random.random() < self.epsilon:
            # Random exploration - prefer movement over waiting
            actions = ["UP", "LEFT", "RIGHT", "WAIT"]
            valid_actions = [a for a in actions if a == "WAIT" or self._can_move(a)]
            # Weight movement actions more heavily
            movement_actions = [a for a in valid_actions if a != "WAIT"]
            if movement_actions and random.random() < 0.8:
                return random.choice(movement_actions)
            return random.choice(valid_actions)
        
        # Q-value based selection
        qs = q_table[state]
        
        # Filter actions by validity
        valid_actions = []
        for action in self.ACTIONS:
            if action in ["UP", "LEFT", "RIGHT"]:
                if self._can_move(action):
                    valid_actions.append(action)
            elif action not in ["DOWN"]:
                valid_actions.append(action)
        
        if not valid_actions:
            return "WAIT"
        
        # Get Q-values for valid actions only
        valid_q_values = {action: qs[action] for action in valid_actions}
        best_value = max(valid_q_values.values())
        best_actions = [a for a, v in valid_q_values.items() if v == best_value]
        
        return random.choice(best_actions)
    
    def _get_direct_action(self, direction):
        """Convert direction to action"""
        direction_map = {
            "N": "UP",
            "S": "WAIT",
            "E": "RIGHT",
            "W": "LEFT"
        }
        return direction_map.get(direction, "WAIT")
    
    def _get_new_pos(self, action):
        """Get resulting position from action"""
        x, y = self.pos
        if action == "UP":
            return (x, y+1)
        elif action == "DOWN":
            return (x, y-1)
        elif action == "LEFT":
            return (x-1, y)
        elif action == "RIGHT":
            return (x+1, y)
        else:
            return (x, y)
    
    def _direct_navigate(self):
        """Direct navigation towards target when assigned - kept for compatibility"""
        my_pos = self.pos
        target_pos = None
        
        # Determine target
        if self.load >= self.capacity * self.p.unload_threshold:
            target_pos = self.model.depot.pos
        elif self.assigned_bin and self.assigned_bin.state in ("Ready", "Servicing"):
            target_pos = self.assigned_bin.pos
        
        if not target_pos:
            return "WAIT"
        
        # If at target, take appropriate action
        if my_pos == target_pos:
            if target_pos == self.model.depot.pos:
                return "DROP"
            else:
                return "PICK"
        
        # Simple pathfinding towards target - try different directions if blocked
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        
        # Priority list of moves to try
        moves = []
        if abs(dx) > abs(dy):
            if dx > 0:
                moves = ["RIGHT", "UP", "LEFT"]
            else:
                moves = ["LEFT", "UP", "RIGHT"]
        else:
            if dy > 0:
                moves = ["UP", "RIGHT" if dx > 0 else "LEFT", "LEFT" if dx > 0 else "RIGHT"]
            else:
                moves = ["RIGHT" if dx > 0 else "LEFT", "LEFT" if dx > 0 else "RIGHT", "UP"]
        
        # Try each move to see if it's valid
        for move in moves:
            if self._can_move(move):
                return move
        
        return "WAIT"

    def _is_intersection(self):
        """Detect if current cell is an intersection"""
        x, y = self.pos
        left = self.model.in_bounds((x-1, y)) and self.model.cell_free((x-1, y))
        right = self.model.in_bounds((x+1, y)) and self.model.cell_free((x+1, y))
        up = self.model.in_bounds((x, y+1)) and self.model.cell_free((x, y+1))
        down = self.model.in_bounds((x, y-1)) and self.model.cell_free((x, y-1))
        return left and right and up and down

    def _can_move(self, action):
        """Check if a movement action is valid"""
        x, y = self.pos
        if action == "UP":
            new_pos = (x, y+1)
        elif action == "DOWN":
            return False
        elif action == "LEFT":
            new_pos = (x-1, y)
        elif action == "RIGHT":
            new_pos = (x+1, y)
        else:
            return True  # Non-movement actions are always "valid"

        if self._is_intersection() and action not in ("LEFT", "RIGHT"):
            return False
        
        # Check bounds first
        if not self.model.in_bounds(new_pos):
            return False
        
        # Check traffic lights at current position
        for tl in self.model.traffic_lights:
            if tl.pos == (x, y) and tl.phase == "R":
                return False
        
        # Check if destination cell is free
        return self.model.cell_free(new_pos)

    def apply_action_enhanced(self, action):
        """Enhanced action application with sophisticated reward system"""
        # Track position history
        current_pos = self.pos
        self.position_history.append(current_pos)
        self.last_positions.append(current_pos)
        self.exploration_map[current_pos] += 1
        
        # Initialize reward with context-aware base penalty
        base_penalty = -0.1
        if self.is_corner():
            base_penalty = -1.0  # Penalty for staying in corners
        
        reward = base_penalty
        done_event = None
        x, y = self.pos

        # Enhanced movement helpers with better collision detection
        def try_move_enhanced(nx, ny):
            nonlocal reward
            if not self.model.in_bounds((nx, ny)):
                reward -= 3  # Boundary penalty
                return False
            
            # Enhanced traffic light handling
            for tl in self.model.traffic_lights:
                if tl.pos == (x, y) and tl.phase == "R" and (nx,ny) != (x,y):
                    reward -= 5  # Red light violation penalty
                    return False
            
            if not self.model.cell_free((nx, ny)):
                reward -= 2  # Obstacle collision penalty
                self.stuck_counter += 1
                return False
            
            # Successful movement
            self.model.grid.move_to(self, (nx, ny))
            self.total_distance += 1
            self.stuck_counter = max(0, self.stuck_counter - 1)
            
            # Reward for leaving corners
            if self.is_corner(current_pos) and not self.is_corner((nx, ny)):
                reward += 3  # Reward escaping corners
            
            # Reward for exploring new areas (but don't penalize too much for revisiting)
            if self.exploration_map[(nx, ny)] == 0:
                reward += 0.5  # Small exploration bonus
            
            return True

        # Set current target for reward calculations
        if self.load >= self.capacity * self.p.unload_threshold:
            self._target_pos = self.model.depot.pos
        elif self.assigned_bin and self.assigned_bin.state in ("Ready","Servicing"):
            self._target_pos = self.assigned_bin.pos
        else:
            self._target_pos = None

        # Execute actions with enhanced logic (allow multiple moves per step)
        if action == "UP":
            for _ in range(self.speed):
                x, y = self.pos
                if not try_move_enhanced(x, y+1):
                    break
        elif action == "DOWN":
            for _ in range(self.speed):
                x, y = self.pos
                if not try_move_enhanced(x, y-1):
                    break
        elif action == "LEFT":
            for _ in range(self.speed):
                x, y = self.pos
                if not try_move_enhanced(x-1, y):
                    break
        elif action == "RIGHT":
            for _ in range(self.speed):
                x, y = self.pos
                if not try_move_enhanced(x+1, y):
                    break
        elif action == "WAIT":
            # Context-dependent waiting penalties
            if self.is_corner():
                reward -= 2  # Penalty for waiting in corners
            else:
                reward -= 0.2  # Small waiting penalty
                
        elif action == "PICK":
            if (self.assigned_bin and 
                self.pos == self.assigned_bin.pos and 
                self.assigned_bin.state in ("Ready","Servicing")):
                vol = min(self.p.pick_amount, self.capacity - self.load, self.assigned_bin.fill)
                if vol > 0:
                    self.load += vol
                    self.assigned_bin.fill -= vol
                    self.assigned_bin.state = "Servicing"
                    reward += 50 + vol * 10  # MASSIVE pickup reward for aggressive collection
                    
                    if self.assigned_bin.fill <= 1e-6:
                        self.assigned_bin.state = "Done"
                        self.collected_bins += 1
                        reward += 200  # HUGE completion bonus - prioritize finishing bins
                        self.efficiency_score += 50
                        print(f"TRUCK {self.id}: Completed bin {self.assigned_bin.id}! Total collected: {self.collected_bins}")
                        # Report done & free task
                        self.model.dispatcher_queue.append(("INFORM-DONE", self, {"bin": self.assigned_bin}))
                        self.assigned_bin = None
                        self.state = "Idle" if self.load == 0 else "NeedUnload"
                else:
                    reward -= 5  # Stronger penalty for failed pickup
            else:
                reward -= 2  # Penalty for invalid pickup
                
        elif action == "DROP":
            if self.pos == self.model.depot.pos and self.load > 0:
                volume_bonus = self.load * 5  # Higher volume bonus
                reward += 100 + volume_bonus  # MASSIVE unload reward
                self.efficiency_score += 20
                self.load = 0.0
                self.state = "Idle"
            else:
                reward -= 5  # Stronger penalty for invalid drop
        
        elif action in ["RETREAT", "EXPLORE"]:
            # These are handled like movement actions
            reward -= 0.1  # Small penalty for special actions

        # Stuck detection and penalty
        if len(set(self.last_positions)) <= 2 and len(self.last_positions) >= 4:
            reward -= 5  # Penalty for being stuck in a loop
            self.stuck_counter += 1
        
        # Update learning parameters with decay every step
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.last_reward = reward
        # Energy usage
        if action in ("UP", "LEFT", "RIGHT"):
            self.energy = max(0, self.energy - self.p.energy_usage * self.speed)
        elif action == "WAIT":
            self.energy = max(0, self.energy - self.p.energy_usage * 0.5)
        elif action in ("PICK", "DROP"):
            self.energy = max(0, self.energy - self.p.energy_usage * 0.2)

        return reward, done_event

    def step(self):
        if not self.active:
            return

        # Track the last action taken
        current_action = None
        
        # Log major actions for debugging
        if len(self.action_log) < 50:
            self.action_log.append({
                "tick": self.model.t,
                "pos": self.pos,
                "assigned_bin": self.assigned_bin.static_id if self.assigned_bin else None,
                "load": round(self.load, 2),
                "corner_timer": self.corner_timer,
                "stuck_counter": self.stuck_counter
            })

        # Enhanced state management
        if self.energy <= self.p.energy_threshold:
            self.state = "NeedCharge"
            self.assigned_bin = None
        elif self.load >= self.capacity * self.p.unload_threshold:
            self.state = "NeedUnload"
        elif self.assigned_bin and self.assigned_bin.state in ("Ready","Servicing"):
            self.state = "Navigating"
        elif not self.assigned_bin:
            self.state = "Patrolling"

        # Enhanced Q-learning loop with multiple Q-tables
        state = self.perceive_enhanced_state()
        action = self.choose_action_enhanced(state)
        
        reward, _ = self.apply_action_enhanced(action)
        new_state = self.perceive_enhanced_state()
        
        # Update appropriate Q-table
        current_q_table = self.select_q_table(state)
        next_q_table = self.select_q_table(new_state)
        
        # Enhanced Q-learning update with experience replay
        best_next = max(next_q_table[new_state].values())
        td_error = reward + self.gamma * best_next - current_q_table[state][action]
        current_q_table[state][action] += self.alpha * td_error
        
        # Cross-table learning (knowledge transfer between contexts)
        if abs(td_error) > 0.5:  # Significant learning
            for q_table in [self.q_navigation, self.q_exploration, self.q_emergency]:
                if q_table != current_q_table:
                    q_table[state][action] += self.alpha * 0.1 * td_error

        # Track last action for logging
        self.last_action = action
        current_action = action

        # Charging behavior
        if self.state == "NeedCharge" and self.pos == self.model.dispatcher.pos:
            self.state = "Charging"
            self.energy = min(self.p.truck_energy_max, self.energy + self.p.charge_rate)
            self.last_action = "CHARGE"
            print(f"TRUCK {self.id}: Charging ({self.energy:.1f}%)")
            if self.energy >= self.p.truck_energy_max:
                self.state = "Idle"
                print(f"TRUCK {self.id}: Battery full")

        # Out of bounds safety
        if not self.model.in_bounds(self.pos):
            print(f"TRUCK {self.id}: went out of bounds and is shut down")
            self.active = False

# -------------------------
# The Model
# -------------------------

class CityWasteModel(ap.Model):
    def setup(self):
        # Grid
        self.grid = ap.Grid(self, (self.p.width, self.p.height), track_empty=True)
        self.t = 0

        # === Cargar JSON de lanes ===
        with open("config_Sim/roadZones.json", "r") as f:
            lane_data = json.load(f)

        self.spawn_points = self._extract_lane_spawns(lane_data["roads"])
        
        # Ensure we have enough spawn points, add random ones if needed
        if len(self.spawn_points) < self.p.n_trucks:
            print(f"WARNING: Only {len(self.spawn_points)} spawn points found, need {self.p.n_trucks}")
            for _ in range(self.p.n_trucks - len(self.spawn_points)):
                rand_x = random.randint(50, self.p.width - 50)
                rand_z = random.randint(50, self.p.height - 50)
                self.spawn_points.append(((rand_x, rand_z), 0))
            print(f"Added random spawn points, total: {len(self.spawn_points)}")

        # Messaging and buffers
        self.dispatcher_queue = deque()
        self.defer_buffer = deque()

        # Create entities with static IDs
        # ID_01 Dispatcher, ID_02 Depot, Trucks ID_10X, Bins ID_20X, TL ID_30X, Obst ID_40X
        self.dispatcher = self._make_agent(Dispatcher, "ID_01", label="Dispatcher")
        self.depot = self._make_agent(Depot, "ID_02", label="Depot")

        # Place depot and dispatcher
        self.grid.add_agents([self.depot], positions=[self.p.depot_pos])
        self.grid.add_agents([self.dispatcher], positions=[self.p.dispatcher_pos])

        # Bins
        # === Cargar JSON de trash bins ===
        with open("config_Sim/trashBinZones.json", "r") as f:
            bin_zone_data = json.load(f)

        # Extraer todos los posibles puntos (hasta 60)
        zone_positions = []
        for z in bin_zone_data["zones"]:
            # Usamos el inicio del collider como posición fija y transformamos las coordenadas
            json_x, json_z = int(z["center_x"]), int(z["center_z"])
            grid_x, grid_z = transform_coordinates(json_x, json_z, self.p.coord_offset_x, self.p.coord_offset_z)
            
            # Verificar que esté dentro de los límites
            if is_valid_grid_position(grid_x, grid_z, self.p.width, self.p.height):
                zone_positions.append((grid_x, grid_z))
            else:
                # Si está fuera de límites, usar posición aleatoria
                rand_x = random.randint(10, self.p.width - 10)
                rand_z = random.randint(10, self.p.height - 10)
                zone_positions.append((rand_x, rand_z))

        # Aleatorizamos los espacios
        random.shuffle(zone_positions)

        # Tomamos solo los primeros n_bins o todos los disponibles
        chosen_positions = zone_positions[:self.p.n_bins]

        # Crear bins
        self.bins = []
        for i in range(1, self.p.n_bins + 1):
            b = self._make_agent(TrashBin, f"ID_2{str(i).zfill(2)}")
            self.bins.append(b)
            self.bins = ap.AgentList(self, self.bins)

        # Asignar posiciones desde chosen_positions
        valid_bin_count = 0
        for b in self.bins:
            if valid_bin_count < len(chosen_positions):
                pos = chosen_positions[valid_bin_count]
                # Validate position is within bounds
                if self.in_bounds(pos):
                    self.grid.add_agents([b], positions=[pos])
                    valid_bin_count += 1
                else:
                    print(f"WARNING: Bin position {pos} is outside bounds, using random position")
                    try:
                        pos = self._rand_free()
                        self.grid.add_agents([b], positions=[pos])
                        valid_bin_count += 1
                    except:
                        # If can't find free position, place at safe location
                        safe_pos = (self.p.width // 2 + valid_bin_count * 2, self.p.height // 2)
                        if self.in_bounds(safe_pos):
                            self.grid.add_agents([b], positions=[safe_pos])
                            valid_bin_count += 1
            else:
                # Not enough valid positions, use random
                try:
                    pos = self._rand_free()
                    self.grid.add_agents([b], positions=[pos])
                except:
                    # Fallback to safe grid position
                    safe_pos = (self.p.width // 2 + valid_bin_count * 2, self.p.height // 2)
                    if self.in_bounds(safe_pos):
                        self.grid.add_agents([b], positions=[safe_pos])
                        valid_bin_count += 1

        # Initialize bin states based on their current fill levels
        for b in self.bins:
            b.notify_ready()

        # Traffic lights
# === Cargar JSON de traffic lights ===
        with open("config_Sim/trafficLights.json", "r") as f:
            tl_data = json.load(f)

        self.traffic_lights = []
        for i, t in enumerate(tl_data["lights"][:self.p.n_tlights], start=1):
            tl = self._make_agent(TrafficLight, f"ID_3{str(i).zfill(2)}")
            self.traffic_lights.append(tl)

        self.traffic_lights = ap.AgentList(self, self.traffic_lights)

        # Asignar posiciones del JSON con transformación de coordenadas
        for tl, data in zip(self.traffic_lights, tl_data["lights"]):
            json_x, json_z = int(data["x"]), int(data["z"])
            grid_x, grid_z = transform_coordinates(json_x, json_z, self.p.coord_offset_x, self.p.coord_offset_z)
            pos = (grid_x, grid_z)
            
            # Validate traffic light position
            if is_valid_grid_position(grid_x, grid_z, self.p.width, self.p.height):
                self.grid.add_agents([tl], positions=[pos])
            else:
                # Use random position within bounds
                rand_x = random.randint(20, self.p.width - 20)
                rand_z = random.randint(20, self.p.height - 20)
                pos = (rand_x, rand_z)
                self.grid.add_agents([tl], positions=[pos])

        # Trucks
        self.trucks = []
        for i in range(1, self.p.n_trucks + 1):
            tr = self._make_agent(GarbageTruck, f"ID_1{str(i).zfill(2)}")
            self.trucks.append(tr)
        self.trucks = ap.AgentList(self, self.trucks)

        # Usar lanes como spawn points (si se acaban, usar random)
        # Trucks orientation
        for i, tr in enumerate(self.trucks):
            if i < len(self.spawn_points):
                pos, angle = self.spawn_points[i]
                # Double-check that spawn position is valid
                if self.in_bounds(pos) and self.cell_free(pos):
                    self.grid.add_agents([tr], positions=[pos])
                    tr.spawn_angle = angle   # guardar ángulo inicial
                else:
                    print(f"WARNING: Spawn point {pos} is invalid, using random position")
                    pos = self._rand_free()
                    self.grid.add_agents([tr], positions=[pos])
                    tr.spawn_angle = 0
            else:
                pos = self._rand_free()
                self.grid.add_agents([tr], positions=[pos])
                tr.spawn_angle = 0

        # Obstacles
        self.obstacles = []
        available_spawns = list(self.spawn_points)  # usar los mismos que los trucks
        random.shuffle(available_spawns)

        for i in range(1, self.p.n_obstacles + 1):
            ob = self._make_agent(DynamicObstacle, f"ID_4{str(i).zfill(2)}")
            self.obstacles.append(ob)

        self.obstacles = ap.AgentList(self, self.obstacles)

        used_positions = {self.depot.pos, self.dispatcher.pos}
        used_positions.update([pos for (pos, ang) in self.spawn_points[:len(self.trucks)]])  # trucks ocupados

        for ob in self.obstacles:
            # buscar un spawn libre
            pos, angle = None, 0
            while available_spawns:
                cand_pos, cand_angle = available_spawns.pop()
                if cand_pos not in used_positions:
                    pos, angle = cand_pos, cand_angle
                    used_positions.add(pos)
                    break
            if pos is None:
                # fallback si no queda ningún spawn libre
                pos = self._rand_free()
                angle = 0
            self.grid.add_agents([ob], positions=[pos])
            ob.spawn_angle = angle

        # Progress tracking
        self._start_time = time.perf_counter()
        self._last_eta_print = -999

        # KPIs
        self.total_bins_done = 0
        
        # History tracking for visualization
        self.history = {'steps': []}

    # ---- helpers ----
    def _extract_lane_spawns(self, roads):
        spawns = []
        for road in roads:
            for lane in road["lanes"]:
                json_x1, json_z1 = lane["start_x"], lane["start_z"]
                json_x2, json_z2 = lane["end_x"], lane["end_z"]

                json_mid_x = (json_x1 + json_x2) / 2
                json_mid_z = (json_z1 + json_z2) / 2
                
                # Transform coordinates to grid space
                grid_x, grid_z = transform_coordinates(json_mid_x, json_mid_z, 
                                                     self.p.coord_offset_x, self.p.coord_offset_z)

                lane_width = lane["width"]
                if lane_width >= self.p.truck_width:
                    angle = lane.get("rot_y", 0)  # usar el rot_y exportado
                    spawn_pos = (grid_x, grid_z)
                    
                    # CRITICAL FIX: Only add spawn points that are within bounds
                    if is_valid_grid_position(grid_x, grid_z, self.p.width, self.p.height):
                        spawns.append((spawn_pos, angle))
                    else:
                        # Use random fallback position within bounds
                        rand_x = random.randint(50, self.p.width - 50)
                        rand_z = random.randint(50, self.p.height - 50)
                        spawns.append(((rand_x, rand_z), angle))
        
        return spawns

    def _extract_lane_positions(self, roads, step=5):
        """Genera múltiples puntos de spawn a lo largo de cada lane"""
        positions = []
        for road in roads:
            for lane in road["lanes"]:
                x1, z1 = lane["start_x"], lane["start_z"]
                x2, z2 = lane["end_x"], lane["end_z"]
                angle = lane.get("rot_y", 0)

                # número de subdivisiones según largo del segmento
                length = int(((x2-x1)**2 + (z2-z1)**2) ** 0.5)
                n_steps = max(1, length // step)

                for i in range(n_steps+1):
                    t = i / n_steps
                    px = x1 + (x2 - x1) * t
                    pz = z1 + (z2 - z1) * t
                    positions.append(((int(px), int(pz)), angle))
        return positions

    def _make_agent(self, cls, static_id, label=None):
        # Create a temporary class that sets static_id before setup
        class TempAgent(cls):
            def __init__(self, model):
                self.static_id = static_id
                if label: self.label = label
                super().__init__(model)
        
        return TempAgent(self)

    def _rand_free(self):
        # Sample a free cell within bounds
        max_attempts = 5000
        for attempt in range(max_attempts):
            pos = (random.randrange(10, self.p.width - 10), random.randrange(10, self.p.height - 10))
            if self.in_bounds(pos) and self.cell_free(pos):
                return pos
        
        # Fallback: find any free position systematically
        for x in range(10, self.p.width - 10):
            for y in range(10, self.p.height - 10):
                pos = (x, y)
                if self.in_bounds(pos) and self.cell_free(pos):
                    return pos
        
        # Last resort: use center of map
        center_pos = (self.p.width // 2, self.p.height // 2)
        if self.in_bounds(center_pos):
            return center_pos
        
        raise RuntimeError("No free cell found for placement")

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.p.width and 0 <= y < self.p.height

    def cell_free(self, pos):
        # Free if no agents at this position (except trucks can share with bins/TLs)
        try:
            # Check if position is in bounds first
            if not self.in_bounds(pos):
                return False
            # Get all agents at this position
            agents_at_pos = [agent for agent in self.grid.agents if self.grid.positions.get(agent) == pos]
            
            # Check for blocking agents (trucks, obstacles, depot)
            blocking_agents = [a for a in agents_at_pos if a.entity_type in ("truck", "obstacle", "depot")]
            return len(blocking_agents) == 0
        except:
            return True  # If there's an error, assume it's free

    def truck_load(self, truck):
        return truck.load

    def _check_collisions(self):
        """Disable trucks that collide"""
        positions = {}
        for tr in self.trucks:
            if not tr.active:
                continue
            pos = tr.pos
            if pos in positions:
                other = positions[pos]
                tr.active = False
                other.active = False
                print(f"Collision detected between {tr.static_id} and {other.static_id} at {pos}")
            else:
                positions[pos] = tr

    # ---- main loop ----

    def step(self):
        self.t += 1

        # Add delay between steps for realistic timing
        if hasattr(self.p, 'step_delay') and self.p.step_delay > 0:
            time.sleep(self.p.step_delay)

        # 1) Environment dynamics
        self.traffic_lights.step()
        self.obstacles.step()
        self.bins.step()

        # 2) Negotiation
        self.dispatcher.step()

        # 3) Trucks move/learn
        self.trucks.step()

        # Collision detection
        self._check_collisions()

        # 4) Collect KPIs
        bins_done_now = sum(1 for b in self.bins if b.state == "Done")
        self.total_bins_done = bins_done_now

        # 5) Record history for visualization (every 10 steps to save memory)
        if self.t % 10 == 0 or self.t == 1:
            self._record_history()

        # 6) Progress / ETA print
        self._print_eta_if_needed()

    def _print_eta_if_needed(self):
        # Print ETA every eta_interval ticks with status info
        if (self.t - self._last_eta_print) >= self.p.eta_interval or self.t == self.p.steps:
            elapsed = time.perf_counter() - self._start_time
            steps_done = self.t
            steps_total = self.p.steps
            per_step = elapsed / max(1, steps_done)
            remaining = steps_total - steps_done
            eta = remaining * per_step
            
            # Add meaningful status info
            ready_bins = sum(1 for b in self.bins if b.state == "Ready")
            servicing_bins = sum(1 for b in self.bins if b.state == "Servicing")
            done_bins = sum(1 for b in self.bins if b.state == "Done")
            total_truck_load = sum(tr.load for tr in self.trucks)
            open_tasks = len(self.dispatcher.open_tasks)
            pending_cfps = len(self.dispatcher.pending_cfp)
            
            print(f"[tick {self.t}/{steps_total}] elapsed={elapsed:.2f}s, ETA={eta:.2f}s")
            print(f"  Bins: {ready_bins} ready, {servicing_bins} servicing, {done_bins} done")
            print(f"  Trucks: total load={total_truck_load:.1f}, tasks: {open_tasks} open, {pending_cfps} bidding")
            self._last_eta_print = self.t

    def _record_history(self):
        """Record current state for visualization history"""
        step_data = {
            "tick": self.t,
            "agents": []
        }
        
        # Record all agents' states
        all_agents = (
            [self.dispatcher, self.depot] + 
            list(self.trucks) + 
            list(self.bins) + 
            list(self.traffic_lights) + 
            list(self.obstacles)
        )
        
        for agent in all_agents:
            # Use the entity_type attribute instead of class name
            agent_type = getattr(agent, 'entity_type', 'unknown')
            
            agent_data = {
                "id": agent.static_id,
                "type": agent_type,
                "x": agent.pos[0],
                "z": agent.pos[1]
            }
            
            # Add specific data based on agent type
            if agent_type == "truck":
                agent_data.update({
                    "load": agent.load,
                    "assigned_bin": agent.assigned_bin.static_id if agent.assigned_bin else None,
                    "state": getattr(agent, 'state', 'Unknown'),
                    "corner_timer": getattr(agent, 'corner_timer', 0),
                    "stuck_counter": getattr(agent, 'stuck_counter', 0),
                    "efficiency_score": getattr(agent, 'efficiency_score', 0),
                    "last_action": getattr(agent, 'last_action', 'WAIT'),
                    "is_in_corner": agent.is_corner() if hasattr(agent, 'is_corner') else False
                })
            elif agent_type == "trash_bin":
                agent_data.update({
                    "fill_level": agent.fill,
                    "ready_for_pickup": agent.state == "Ready",
                    "state": agent.state
                })
            elif agent_type == "traffic_light":
                agent_data.update({
                    "phase": agent.phase
                })
            
            step_data["agents"].append(agent_data)
        
        self.history['steps'].append(step_data)

    def end(self):
        # Enhanced JSON dump with detailed state information
        state = {
            "tick": self.t,
            "simulation_stats": {
                "total_bins_done": self.total_bins_done,
                "total_collected": sum(tr.collected_bins for tr in self.trucks),
                "total_distance": sum(tr.total_distance for tr in self.trucks),
                "open_tasks": len(self.dispatcher.open_tasks),
                "pending_cfps": len(self.dispatcher.pending_cfp)
            },
            "agents": []
        }

        def add_agent(agent_obj):
            x, z = agent_obj.pos_xz()
            agent_data = {
                "id": agent_obj.static_id,
                "type": agent_obj.entity_type,
                "label": getattr(agent_obj, "label", agent_obj.entity_type),
                "x": x,
                "z": z
            }
            
            # Add type-specific state information
            if hasattr(agent_obj, 'state'):
                agent_data["state"] = agent_obj.state
            
            if agent_obj.entity_type == "truck":
                agent_data.update({
                    "load": round(agent_obj.load, 2),
                    "capacity": agent_obj.capacity,
                    "collected_bins": agent_obj.collected_bins,
                    "total_distance": agent_obj.total_distance,
                    "assigned_bin": agent_obj.assigned_bin.static_id if agent_obj.assigned_bin else None,
                    "last_action": getattr(agent_obj, 'last_action', 'WAIT'),
                    "action_log": agent_obj.action_log[-10:],  # Last 10 actions for debugging
                    "rotation_y": agent_obj.spawn_angle if hasattr(agent_obj, "spawn_angle") else 0,
                    # Enhanced Q-Learning metrics
                    "corner_timer": getattr(agent_obj, 'corner_timer', 0),
                    "stuck_counter": getattr(agent_obj, 'stuck_counter', 0),
                    "efficiency_score": getattr(agent_obj, 'efficiency_score', 0),
                    "alpha": round(agent_obj.alpha, 3),
                    "epsilon": round(agent_obj.epsilon, 3),
                    "last_reward": getattr(agent_obj, 'last_reward', 0),
                    "exploration_coverage": len(agent_obj.exploration_map),
                    "is_in_corner": agent_obj.is_corner()
                })
            elif agent_obj.entity_type == "trash_bin":
                agent_data.update({
                    "fill_level": round(agent_obj.fill, 2),
                    "threshold": agent_obj.threshold,
                    "ready_for_pickup": agent_obj.state == "Ready"
                })
            elif agent_obj.entity_type == "traffic_light":
                agent_data.update({
                    "phase": agent_obj.phase,
                    "cycle_position": agent_obj.t
                })
            elif agent_obj.entity_type == "obstacle":
                agent_data.update({
                    "rotation_y": agent_obj.spawn_angle if hasattr(agent_obj, "spawn_angle") else 0
                })

            state["agents"].append(agent_data)

        add_agent(self.dispatcher)
        add_agent(self.depot)
        for group in [self.trucks, self.bins, self.traffic_lights, self.obstacles]:
            for a in group:
                add_agent(a)

        # Create directories if they don't exist
        import os
        os.makedirs("results/simulation_data", exist_ok=True)
        
        out_path = "results/simulation_data/mas_final_state.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Save history for visualization
        history_path = "results/simulation_data/simulation_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Final stats
        total_collect = sum(tr.collected_bins for tr in self.trucks)
        total_distance = sum(tr.total_distance for tr in self.trucks)
        print("\n=== Simulation finished ===")
        print(f"Ticks: {self.t}")
        print(f"Bins Done (by state): {self.total_bins_done}")
        print(f"Bins Collected (truck counters): {total_collect}")
        print(f"Fleet distance traveled: {total_distance}")
        print(f"Output JSON: {out_path}")
        print(f"History JSON: {history_path}")

# -------------------------
# Parameters & run
# -------------------------

DEFAULT_PARAMS = {
    # Grid/world - Adjusted to accommodate JSON coordinate ranges
    "width": 500,    # Increased to accommodate coordinate range
    "height": 400,   # Increased to accommodate coordinate range
    "coord_offset_x": 260,  # Offset to map JSON X coords (-260 to +60) to grid (0 to 320)
    "coord_offset_z": 120,  # Offset to map JSON Z coords (-120 to +200) to grid (0 to 320)
    "steps": 3000,  # Extended run for better performance

    # Entities - MORE BINS FOR BETTER PERFORMANCE
    "n_trucks": 8,
    "n_bins": 40,    # DOUBLED from 20 to 40 bins for more opportunities
    "n_tlights": 15,  # More traffic lights
    "n_obstacles": 10, # More obstacles

    # Placements (using transformed coordinates)
    "depot_pos": (151, 299),        # Transformed and clamped from (-109, 299)
    "dispatcher_pos": (150, 274),   # Transformed and clamped from (-110, 274)

    # Bins - AGGRESSIVE SETTINGS FOR HIGH PERFORMANCE  
    "bin_init_min": 0.8,     # Start bins much closer to threshold
    "bin_init_max": 1.5,     # Start some bins well above threshold 
    "bin_threshold": 0.8,
    "bin_fill_rate": 0.05,   # Much faster fill rate so bins get full quickly

    # Traffic lights
    "tl_cycle": 8,           # Shorter cycles for faster movement
    "tl_green": 6,           # Longer green phases

    # Obstacles
    "obstacle_move_prob": 0.05, # Slower moving obstacles for easier navigation

    # Trucks & operations - OPTIMIZED FOR MAXIMUM EFFICIENCY
    "truck_capacity": 4.0,      # Higher capacity
    "truck_speed": 2,           # Move two cells per step for faster travel
    "pick_amount": 1.0,         # Much larger pick amount for faster collection
    "unload_threshold": 0.6,    # Lower threshold - unload sooner
    "truck_energy_max": 100,
    "energy_usage": 1,
    "charge_rate": 10,
    "energy_threshold": 10,

    # Contract Net / Dispatcher
    "cfp_timeout": 4,
    "fairness_alpha": 0.25,  # fairness cost weight

    # Q-learning - MAXIMUM PERFORMANCE SETTINGS
    "q_alpha": 0.8,          # High learning rate for rapid adaptation
    "q_gamma": 0.98,         # Very high discount factor for long-term planning
    "q_epsilon": 0.1,        # Low exploration - focus on exploitation
    "corner_margin": 8,      # Smaller corner margin for more navigation space

    # Misc
    "eta_interval": 100,     # Print ETA less frequently
    "step_delay": 0,

    # Truck dimensions (for spawn checks)
    "truck_width": 3.626759
}

if __name__ == "__main__":
    params = DEFAULT_PARAMS
    
    # Use the DEFAULT_PARAMS settings (6-minute simulation with your custom values)
    # No overrides needed - using your DEFAULT_PARAMS configuration:
    # Simulation uses DEFAULT_PARAMS
    
    model = CityWasteModel(params)
    results = model.run()
