# mas_waste_qlearning_agentpy.py
# Multi-Agent Waste Collection with Contract Net + Q-Learning (AgentPy)
# Author: You (Santiago & team)
# Notes:
# - MAS runs in discrete ticks (Unity can render continuous motion from JSON snapshots).
# - Negotiation: Dispatcher issues CFP, trucks PROPOSE, Dispatcher AWARDs (Contract Net).
# - Trucks learn navigation/operations via Q-Learning.
# - Writes final JSON with static IDs and (x,z) positions.
# - Prints ETA during sim and final KPIs at the end.

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
    ACTIONS = ("UP","DOWN","LEFT","RIGHT","WAIT","PICK","DROP")

    def setup(self):
        super().setup()
        self.entity_type = "truck"
        # Capabilities
        self.capacity = self.p.truck_capacity
        self.load = 0.0
        self.speed = 1  # one cell per step

        # Task & motion
        self.assigned_bin = None
        self.state = "Idle"
        self.total_distance = 0
        self.last_action = "WAIT"  # Track last action taken
        self.action_log = []  # Track all actions for debugging

        # Q-learning params
        self.alpha = self.p.q_alpha
        self.gamma = self.p.q_gamma
        self.epsilon = self.p.q_epsilon
        self.q = defaultdict(lambda: {a: 0.0 for a in self.ACTIONS})

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

    # ---------------- Q-learning core ---------------------

    def perceive_state(self):
        x, y = self.pos
        # Relative direction to target bin (or depot if unloading is needed)
        target = None
        if self.load >= self.capacity * self.p.unload_threshold:
            target = self.model.depot.pos
        elif self.assigned_bin and self.assigned_bin.state in ("Ready","Servicing"):
            target = self.assigned_bin.pos

        def dir_to(a, b):
            if b is None: return "NONE"
            dx, dy = b[0]-a[0], b[1]-a[1]
            if abs(dx) > abs(dy):
                return "E" if dx > 0 else "W"
            elif abs(dy) > 0:
                return "N" if dy > 0 else "S"
            else:
                return "HERE"

        # Traffic light at my cell?
        tl_here = "G"
        for tl in self.model.traffic_lights:
            if tl.pos == self.pos:
                tl_here = tl.phase
                break

        has_load = 1 if self.load > 0 else 0
        assigned = 1 if self.assigned_bin else 0
        return (x, y, dir_to((x,y), target), tl_here, has_load, assigned)

    def choose_action(self, s):
        # Prioritize direct navigation when we have a clear target
        if self.assigned_bin:
            action = self._direct_navigate()
            if action != "WAIT":
                return action
        
        # Use Q-learning only for exploration when no clear target
        if random.random() < self.epsilon * 0.1:  # Reduce randomness significantly
            return random.choice(self.ACTIONS)
        qs = self.q[s]
        # argmax with tie-break
        best = max(qs.items(), key=lambda kv: kv[1])[1]
        cands = [a for a,v in qs.items() if v == best]
        return random.choice(cands)
    
    def _direct_navigate(self):
        """Direct navigation towards target when assigned"""
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
                moves = ["RIGHT", "UP" if dy > 0 else "DOWN", "DOWN" if dy > 0 else "UP", "LEFT"]
            else:
                moves = ["LEFT", "UP" if dy > 0 else "DOWN", "DOWN" if dy > 0 else "UP", "RIGHT"]
        else:
            if dy > 0:
                moves = ["UP", "RIGHT" if dx > 0 else "LEFT", "LEFT" if dx > 0 else "RIGHT", "DOWN"]
            else:
                moves = ["DOWN", "RIGHT" if dx > 0 else "LEFT", "LEFT" if dx > 0 else "RIGHT", "UP"]
        
        # Try each move to see if it's valid
        for move in moves:
            if self._can_move(move):
                return move
        
        return "WAIT"
    
    def _can_move(self, action):
        """Check if a movement action is valid"""
        x, y = self.pos
        if action == "UP":
            new_pos = (x, y+1)
        elif action == "DOWN":
            new_pos = (x, y-1)
        elif action == "LEFT":
            new_pos = (x-1, y)
        elif action == "RIGHT":
            new_pos = (x+1, y)
        else:
            return True  # Non-movement actions are always "valid"
        
        # Check bounds
        if not self.model.in_bounds(new_pos):
            return False
        
        # Check traffic lights
        for tl in self.model.traffic_lights:
            if tl.pos == (x, y) and tl.phase == "R":
                return False
        
        # Check if cell is free
        return self.model.cell_free(new_pos)

    def apply_action(self, a):
        # Track the last action taken
        self.last_action = a
        
        # Log major actions for debugging
        if a in ["PICK", "DROP"] or len(self.action_log) < 50:
            self.action_log.append({
                "tick": self.model.t,
                "action": a,
                "pos": self.pos,
                "assigned_bin": self.assigned_bin.static_id if self.assigned_bin else None,
                "load": round(self.load, 2)
            })
        
        reward = -0.5  # small step penalty to encourage efficiency
        done_event = None

        x, y = self.pos

        # Movement helpers
        def try_move(nx, ny):
            nonlocal reward
            if not self.model.in_bounds((nx, ny)):
                reward -= 2
                return False
            # Red light penalty if present
            for tl in self.model.traffic_lights:
                if tl.pos == (x, y) and tl.phase == "R" and (nx,ny)!=(x,y):
                    reward -= 3  # red-cross attempt penalty
                    return False
            if not self.model.cell_free((nx, ny)):
                reward -= 2  # obstacle or collision
                return False
            # Move
            self.model.grid.move_to(self, (nx, ny))
            self.total_distance += 1
            return True

        # Execute
        if a == "UP":    success = try_move(x, y+1)
        elif a == "DOWN": success = try_move(x, y-1)
        elif a == "LEFT": success = try_move(x-1, y)
        elif a == "RIGHT": success = try_move(x+1, y)
        elif a == "WAIT":
            reward -= 0.1
        elif a == "PICK":
            if self.assigned_bin and self.pos == self.assigned_bin.pos and self.assigned_bin.state in ("Ready","Servicing"):
                vol = min(self.p.pick_amount, self.capacity - self.load, self.assigned_bin.fill)
                if vol > 0:
                    self.load += vol
                    self.assigned_bin.fill -= vol
                    self.assigned_bin.state = "Servicing"
                    reward += 5
                    if self.assigned_bin.fill <= 1e-6:
                        self.assigned_bin.state = "Done"
                        self.collected_bins += 1
                        reward += 10
                        # Report done & free task (INFORM-DONE)
                        self.model.dispatcher_queue.append(("INFORM-DONE", self, {"bin": self.assigned_bin}))
                        self.assigned_bin = None
                        self.state = "Idle" if self.load == 0 else "NeedUnload"
                else:
                    reward -= 0.5
            else:
                reward -= 0.5
        elif a == "DROP":
            if self.pos == self.model.depot.pos and self.load > 0:
                reward += 8 + self.load  # bigger reward for unloading volume
                self.load = 0.0
                self.state = "Idle"
            else:
                reward -= 0.5

        return reward, done_event

    def step(self):
        # Decide high-level state
        if self.load >= self.capacity * self.p.unload_threshold:
            self.state = "NeedUnload"
        elif self.assigned_bin and self.assigned_bin.state in ("Ready","Servicing"):
            self.state = "Navigating"
        elif not self.assigned_bin:
            self.state = "Patrolling"

        # Q-learning loop (1 action per tick)
        s = self.perceive_state()
        a = self.choose_action(s)
        r, _ = self.apply_action(a)
        s2 = self.perceive_state()
        # Update
        best_next = max(self.q[s2].values())
        self.q[s][a] += self.alpha * (r + self.gamma * best_next - self.q[s][a])

# -------------------------
# The Model
# -------------------------

class CityWasteModel(ap.Model):
    def setup(self):
        # Grid
        self.grid = ap.Grid(self, (self.p.width, self.p.height), track_empty=True)
        self.t = 0

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
        self.bins = []
        for i in range(1, self.p.n_bins + 1):
            b = self._make_agent(TrashBin, f"ID_2{str(i).zfill(2)}")
            self.bins.append(b)
        self.bins = ap.AgentList(self, self.bins)
        self.grid.add_agents(self.bins, positions=[self._rand_free() for _ in self.bins])
        
        # Notify dispatcher of initially ready bins
        for b in self.bins:
            b.notify_ready()

        # Traffic lights
        self.traffic_lights = []
        for i in range(1, self.p.n_tlights + 1):
            tl = self._make_agent(TrafficLight, f"ID_3{str(i).zfill(2)}")
            self.traffic_lights.append(tl)
        self.traffic_lights = ap.AgentList(self, self.traffic_lights)
        self.grid.add_agents(self.traffic_lights, positions=[self._rand_free() for _ in self.traffic_lights])

        # Obstacles
        self.obstacles = []
        for i in range(1, self.p.n_obstacles + 1):
            ob = self._make_agent(DynamicObstacle, f"ID_4{str(i).zfill(2)}")
            self.obstacles.append(ob)
        self.obstacles = ap.AgentList(self, self.obstacles)
        self.grid.add_agents(self.obstacles, positions=[self._rand_free() for _ in self.obstacles])

        # Trucks
        self.trucks = []
        for i in range(1, self.p.n_trucks + 1):
            tr = self._make_agent(GarbageTruck, f"ID_1{str(i).zfill(2)}")
            self.trucks.append(tr)
        self.trucks = ap.AgentList(self, self.trucks)
        self.grid.add_agents(self.trucks, positions=[self._rand_free() for _ in self.trucks])

        # Progress tracking
        self._start_time = time.perf_counter()
        self._last_eta_print = -999

        # KPIs
        self.total_bins_done = 0
        
        # History tracking for visualization
        self.history = {'steps': []}

    # ---- helpers ----

    def _make_agent(self, cls, static_id, label=None):
        # Create a temporary class that sets static_id before setup
        class TempAgent(cls):
            def __init__(self, model):
                self.static_id = static_id
                if label: self.label = label
                super().__init__(model)
        
        return TempAgent(self)

    def _rand_free(self):
        # Sample a free cell
        for _ in range(5000):
            pos = (random.randrange(self.p.width), random.randrange(self.p.height))
            if self.cell_free(pos):
                return pos
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
                    "state": getattr(agent, 'state', 'Unknown')
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
                    "action_log": agent_obj.action_log[-10:]  # Last 10 actions for debugging
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
    # Grid/world
    "width": 20,
    "height": 14,
    "steps": 3600,  # Increased for 6-minute simulation (3600 steps)

    # Entities - Adjust number of bins here
    "n_trucks": 5,
    "n_bins": 20,    # Increased from 6 to 20 bins for more work
    "n_tlights": 10,  # Fewer traffic lights 
    "n_obstacles": 15, # Fewer obstacles

    # Placements
    "depot_pos": (1, 1),
    "dispatcher_pos": (0, 0),

    # Bins
    "bin_init_min": 0.6,     # Start bins closer to threshold
    "bin_init_max": 1.2,     # Some bins start above threshold
    "bin_threshold": 0.8,
    "bin_fill_rate": 0.02,   # Faster fill rate so bins get full

    # Traffic lights
    "tl_cycle": 12,          # ticks
    "tl_green": 7,

    # Obstacles
    "obstacle_move_prob": 0.1, # Slower moving obstacles

    # Trucks & operations
    "truck_capacity": 3.0,
    "pick_amount": 0.5,      # Smaller pick amount for more realistic collection
    "unload_threshold": 0.8, # when to aim for depot

    # Contract Net / Dispatcher
    "cfp_timeout": 4,
    "fairness_alpha": 0.25,  # fairness cost weight

    # Q-learning
    "q_alpha": 0.5,
    "q_gamma": 0.90,
    "q_epsilon": 0.05,       # Much less random exploration

    # Misc
    "eta_interval": 100,     # Print ETA less frequently
    "step_delay": 0.1,       # Add delay between steps (0.1 seconds per step for 3-min simulation)
}

if __name__ == "__main__":
    params = DEFAULT_PARAMS
    
    # Use the DEFAULT_PARAMS settings (6-minute simulation with your custom values)
    # No overrides needed - using your DEFAULT_PARAMS configuration:
    # - 3600 steps (6 minutes)
    # - 20 bins
    # - 5 trucks  
    # - 10 traffic lights
    # - 15 obstacles
    
    model = CityWasteModel(params)
    results = model.run()
