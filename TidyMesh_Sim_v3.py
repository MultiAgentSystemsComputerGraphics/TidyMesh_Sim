#   /$$$$$$$$ /$$       /$$           /$$      /$$                     /$$      
#  |__  $$__/|__/      | $$          | $$$    /$$$                    | $$      
#     | $$    /$$  /$$$$$$$ /$$   /$$| $$$$  /$$$$  /$$$$$$   /$$$$$$$| $$$$$$$ 
#     | $$   | $$ /$$__  $$| $$  | $$| $$ $$/$$ $$ /$$__  $$ /$$_____/| $$__  $$
#     | $$   | $$| $$  | $$| $$  | $$| $$  $$$| $$| $$$$$$$$|  $$$$$$ | $$  \ $$
#     | $$   | $$| $$  | $$| $$  | $$| $$\  $ | $$| $$_____/ \____  $$| $$  | $$
#     | $$   | $$|  $$$$$$$|  $$$$$$$| $$ \/  | $$|  $$$$$$$ /$$$$$$$/| $$  | $$
#     |__/   |__/ \_______/ \____  $$|__/     |__/ \_______/|_______/ |__/  |__/    v3
#                           /$$  | $$                                           
#                          |  $$$$$$/                                           
#                           \______/                                            

# TIDYMESH SIMULATION v3
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
import json, math, random, time, os
from collections import defaultdict, deque

# -------------------------
# Small helpers
# -------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def xform(json_x, json_z, offx, offz):
    return int(round(json_x + offx)), int(round(json_z + offz))

def in_bounds(pos, W, H):
    x, y = pos
    return 0 <= x < W and 0 <= y < H

def bresenham(a, b):
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1-x0), -abs(y1-y0)
    sx, sy = (1 if x0<x1 else -1), (1 if y0<y1 else -1)
    err, x, y = dx+dy, x0, y0
    out = []
    while True:
        out.append((x, y))
        if x == x1 and y == y1: break
        e2 = 2*err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy
    return out

def build_road_graph(roads, offx, offz, W, H, thickness=2):
    """Rasterize lanes with Manhattan thickness so crossings connect."""
    mask = set()
    th = max(0, int(thickness))
    for r in roads:
        for ln in r["lanes"]:
            a = xform(ln["start_x"], ln["start_z"], offx, offz)
            b = xform(ln["end_x"],   ln["end_z"],   offx, offz)
            for (px, py) in bresenham(a, b):
                if not in_bounds((px, py), W, H): continue
                for dx in range(-th, th+1):
                    rem = th - abs(dx)
                    for dy in range(-rem, rem+1):
                        q = (px+dx, py+dy)
                        if in_bounds(q, W, H): mask.add(q)
    nbrs = {}
    for x, y in mask:
        ns = []
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if (nx, ny) in mask: ns.append((nx, ny))
        nbrs[(x, y)] = tuple(ns)
    return mask, nbrs

def snap_to_nearest(cell, road_mask, search=12):
    if cell in road_mask: return cell
    x0, y0 = cell
    best, bestd = None, 1e9
    for r in range(1, search+1):
        for dx in range(-r, r+1):
            for dy in (-r, r):
                c = (x0+dx, y0+dy)
                if c in road_mask:
                    d = abs(dx)+abs(dy)
                    if d < bestd: best, bestd = c, d
        for dy in range(-r+1, r):
            for dx in (-r, r):
                c = (x0+dx, y0+dy)
                if c in road_mask:
                    d = abs(dx)+abs(dy)
                    if d < bestd: best, bestd = c, d
        if best is not None: return best
    return cell

def bfs_first_step(src, dst, nbrs):
    if src == dst: return src
    q, seen = deque([src]), {src: None}
    while q:
        u = q.popleft()
        for v in nbrs.get(u, ()):
            if v not in seen:
                seen[v] = u
                if v == dst:
                    cur, prev = v, u
                    while seen[prev] is not None:
                        cur, prev = prev, seen[prev]
                    return cur
                q.append(v)
    return src

# -------------------------
# Base agent
# -------------------------
class Base(ap.Agent):
    def setup(self):
        assert hasattr(self, "static_id")
        self.kind = getattr(self, "kind", "entity")
    @property
    def pos(self): return tuple(self.model.grid.positions[self])
    def pos_xz(self): x,y = self.pos; return int(x), int(y)

# -------------------------
# Depot / Light / Bin / Obstacle
# -------------------------
class Depot(Base):
    def setup(self): super().setup(); self.kind = "depot"

class TrafficLight(Base):
    def setup(self):
        super().setup(); self.kind = "light"
        self.cycle = self.p.tl_cycle; self.green = self.p.tl_green
        self.t = 0; self.phase = "G"
    def step(self):
        self.t = (self.t + 1) % self.cycle
        self.phase = "G" if self.t < self.green else "R"

class TrashBin(Base):
    def setup(self):
        super().setup(); self.kind = "bin"
        self.fill = random.uniform(self.p.bin_init_min, self.p.bin_init_max)
        self.th = self.p.bin_threshold
        self.state = "Ready" if self.fill >= self.th else "Idle"
    def step(self):
        if self.state == "Idle":
            self.fill = clamp(self.fill + self.p.bin_fill_rate, 0, 1.8)
            if self.fill >= self.th:
                self.state = "Ready"
                self.model.dispatcher_queue.append(("READY", self, {"vol": self.fill}))
    def notify_ready(self):
        if self.state == "Ready":
            self.model.dispatcher_queue.append(("READY", self, {"vol": self.fill}))

class RoadObstacle(Base):
    """Road-aware random walker that blocks cells."""
    def setup(self):
        super().setup(); self.kind = "obstacle"
        self.move_prob = getattr(self.p, "obstacle_move_prob", 0.05)
        self.obey_lights = getattr(self.p, "obstacle_obey_lights", True)
        self.spawn_angle = 0  # for JSON compatibility
    def neighbors(self, cell): return self.model.road_nbrs.get(cell, ())
    def step(self):
        # optionally wait on red if the cell is a light
        if self.obey_lights and self.model.blocked_by_red(self.pos):
            return
        if random.random() >= self.move_prob: return
        # try random neighbor that is free
        cand = list(self.neighbors(self.pos))
        random.shuffle(cand)
        for q in cand:
            if self.model.cell_free(q):
                self.model.grid.move_to(self, q)
                break

# -------------------------
# Dispatcher (defensive + fallback award)
# -------------------------
class Dispatcher(Base):
    def setup(self):
        super().setup()
        self.kind = "dispatcher"
        self.open = set()
        self.pending = {}   # bin -> {bids: {truck: cost}, tick: int}
        self.awarded = {}   # bin -> truck
        self.ledger = defaultdict(int)

    def _fallback_assign(self, b):
        cand = []
        for tr in self.model.trucks:
            if getattr(tr, "kind", None) != "truck": continue
            if getattr(tr, "assigned", None) is not None: continue
            dep = self.model.depot.pos
            need = (manhattan(tr.pos, b.pos) + manhattan(b.pos, dep)) * self.p.energy_per_move + self.p.energy_reserve
            if tr.energy >= need:
                cand.append((manhattan(tr.pos, b.pos), tr))
        if cand:
            cand.sort(key=lambda x: x[0]); return cand[0][1]
        return None

    def step(self):
        q = self.model.dispatcher_queue
        msgs = list(q); q.clear()

        for msg, sender, payload in msgs:
            if msg == "READY": self.open.add(sender)
            elif msg == "PROPOSE":
                b = payload["bin"]
                if b in self.pending: self.pending[b]["bids"][sender] = payload["cost"]
            elif msg == "INFORM-DONE":
                b = payload["bin"]
                self.open.discard(b); self.awarded.pop(b, None); self.pending.pop(b, None)
            elif msg == "INFORM-FAIL":
                b = payload["bin"]
                self.open.add(b); self.awarded.pop(b, None); self.pending.pop(b, None)

        for b in list(self.open):
            if b not in self.pending and b not in self.awarded:
                self.pending[b] = {"bids": {}, "tick": self.model.t}
                fleet = [tr for tr in self.model.trucks
                         if getattr(tr, "kind", None) == "truck" and hasattr(tr, "receive_cfp")]
                for tr in fleet: tr.receive_cfp(b)

        timeout = self.p.cfp_timeout
        for b, info in list(self.pending.items()):
            if (self.model.t - info["tick"]) >= timeout or len(info["bids"]) >= len(getattr(self.model, "trucks", [])):
                bids = {tr:c for tr,c in info["bids"].items() if getattr(tr, "assigned", None) is None}
                chosen = None
                if bids:
                    scored = [(c + 0.2*self.ledger[tr], tr) for tr, c in bids.items()]
                    scored.sort(key=lambda x: x[0]); chosen = scored[0][1]
                else:
                    chosen = self._fallback_assign(b)
                if chosen:
                    self.awarded[b] = chosen; self.ledger[chosen] += 1
                    if hasattr(chosen, "receive_award"): chosen.receive_award(b)
                self.pending.pop(b, None)

# -------------------------
# Truck (energy + road-only + simple Q + opportunistic pick)
# -------------------------
class Truck(Base):
    ACTIONS = ("FORWARD","LEFT","RIGHT","PICK","DROP","CHARGE","WAIT")
    def setup(self):
        super().setup(); self.kind = "truck"
        self.cap = getattr(self.p, "truck_capacity", 4.0)
        self.load = 0.0
        self.energy = self.p.energy_max
        self.assigned = None
        self.state = "Idle"
        self.total_distance = 0
        self.collected_bins = 0
        self.alpha = self.p.q_alpha; self.gamma = self.p.q_gamma; self.eps = self.p.q_epsilon
        self.Q = defaultdict(lambda: {a:0.0 for a in self.ACTIONS})
        self.last_action = "WAIT"; self.action_log = []; self.last_reward = 0.0
        self.visited_cells = set(); self.spawn_angle = 0
    def receive_cfp(self, b):
        if self.assigned is not None or self.load >= self.cap*0.95: return
        dep = self.model.depot.pos
        step1 = manhattan(self.pos, b.pos); step2 = manhattan(b.pos, dep)
        need = (step1 + step2) * self.p.energy_per_move + self.p.energy_reserve
        if need > self.energy: return
        cost = step1 + (self.load/self.cap)*2.0
        self.model.dispatcher_queue.append(("PROPOSE", self, {"bin": b, "cost": cost}))
    def receive_award(self, b):
        if self.assigned is None: self.assigned = b; self.state = "ToBin"
    def neighbors(self, cell): return self.model.road_nbrs.get(cell, ())
    def at_intersection(self, cell=None): c = self.pos if cell is None else cell; return len(self.neighbors(c)) >= 3
    def plan_next_step(self, target):
        if target is None: return None
        nxt = self.model.next_step_cached(self.pos, target)
        return nxt if nxt != self.pos else None
    def q_state(self, target):
        W,H = self.p.width, self.p.height; x,y = self.pos
        bx,by = min(9, x*10//max(1,W)), min(9, y*10//max(1,H))
        on_int = 1 if self.at_intersection() else 0
        has_load = min(3, int(self.load / max(1e-6, self.cap) * 4))
        eb = min(3, int(self.energy / max(1, self.p.energy_max) * 4))
        tag = 0 if target is None else (1 if target == self.model.depot.pos else 2)
        return (bx, by, on_int, has_load, eb, tag)
    def reward_for(self, action, moved, reached_target, did_pick, did_drop, did_charge):
        r = -0.02
        if moved: r += 0.05
        if did_pick: r += 6.0
        if did_drop: r += 12.0 + 1.5*self.load
        if did_charge: r += 3.0
        if reached_target: r += 1.0
        if self.energy < self.p.energy_per_move: r -= 10.0
        if action == "PICK" and not did_pick: r -= 2.5
        if action == "DROP" and not did_drop: r -= 3.0
        if action == "CHARGE" and not did_charge: r -= 1.0
        return r
    def step(self):
        self.visited_cells.add(self.pos)
        depot = self.model.depot.pos; target = None
        must_depot = (
            self.energy <= self.p.low_energy or
            self.load >= self.cap*self.p.unload_threshold or
            (self.assigned and self.energy <= (manhattan(self.pos, self.assigned.pos)+manhattan(self.assigned.pos, depot))*self.p.energy_per_move + self.p.energy_reserve)
        )
        if must_depot: target = depot; self.state = "ToDepot"
        elif self.assigned and self.assigned.state in ("Ready","Servicing"):
            target = self.assigned.pos; self.state = "ToBin"
        else:
            ready = [b for b in self.model.bins if b.state == "Ready"]
            if ready:
                tgt = min(ready, key=lambda b: manhattan(self.pos, b.pos))
                target = tgt.pos; self.state = "Patrol"
            else:
                target = depot; self.state = "LoiterDepot"

        light_block = self.model.blocked_by_red(self.pos)
        moved=False; did_pick=did_drop=did_charge=False; reached=False; action_for_log="WAIT"

        # Act if at target
        if target and self.pos == target:
            reached=True
            if target == depot:
                if self.load > 0: self.load = 0.0; did_drop=True; action_for_log="DROP"
                if self.energy < self.p.energy_max: self.energy = clamp(self.energy + self.p.charge_rate, 0, self.p.energy_max); did_charge=True; action_for_log="CHARGE"
            else:
                b=None
                if self.assigned and self.assigned.pos == target: b=self.assigned
                else:
                    for c in self.model.bins_by_pos.get(self.pos, []):
                        if c.state in ("Ready","Servicing"): b=c; break
                if b and self.load < self.cap and b.fill > 0:
                    take = min(self.p.pick_amount, self.cap-self.load, b.fill)
                    self.load += take; b.fill -= take
                    b.state = "Servicing" if b.fill > 1e-6 else "Done"
                    did_pick=True; action_for_log="PICK"
                    if b.state == "Done":
                        self.model.dispatcher_queue.append(("INFORM-DONE", self, {"bin": b}))
                        if self.assigned is b: self.assigned = None
                        self.model.kpis_bins_done += 1; self.collected_bins += 1

        # Move
        next_cell = self.plan_next_step(target)
        if next_cell and not light_block and self.model.cell_free(next_cell):
            if self.energy >= self.p.energy_per_move:
                self.model.grid.move_to(self, next_cell)
                self.total_distance += 1; self.energy -= self.p.energy_per_move
                moved=True; 
                if action_for_log == "WAIT": action_for_log="FORWARD"

        # Q-learning update
        learn_now = self.at_intersection() or moved or reached or did_pick or did_drop or did_charge
        if learn_now:
            s = self.q_state(target)
            if self.at_intersection():
                if random.random() < self.eps: a = random.choice(self.ACTIONS)
                else:
                    qrow = self.Q[s]; best = max(qrow.values()); bestA = [k for k,v in qrow.items() if v==best]; a=random.choice(bestA)
            else:
                a = "FORWARD" if moved else action_for_log
            r = self.reward_for(a, moved, reached, did_pick, did_drop, did_charge)
            s2 = self.q_state(target); best_next = max(self.Q[s2].values())
            self.Q[s][a] += self.alpha * (r + self.gamma*best_next - self.Q[s][a])
            self.last_action = a; self.last_reward = r
        else:
            self.last_action = action_for_log

        self.action_log.append({"tick": self.model.t, "pos": self.pos, "load": round(self.load, 2),
                                "energy": round(self.energy, 2), "state": self.state, "last_action": self.last_action})
        if len(self.action_log) > 10: self.action_log = self.action_log[-10:]

# -------------------------
# Model
# -------------------------
class CityWasteV2(ap.Model):
    def setup(self):
        self.t = 0
        self.grid = ap.Grid(self, (self.p.width, self.p.height), track_empty=True)
        self.dispatcher_queue = deque()

        bp = getattr(self.p, "base_path", "config_Sim")
        with open(os.path.join(bp, "roadZones.json"), "r") as f: roads = json.load(f)["roads"]
        with open(os.path.join(bp, "trafficLights.json"), "r") as f: lights = json.load(f)["lights"]
        with open(os.path.join(bp, "trashBinZones.json"), "r") as f: zones = json.load(f)["zones"]

        thickness = getattr(self.p, "road_thickness", 2)
        self.road_mask, self.road_nbrs = build_road_graph(
            roads, self.p.coord_offset_x, self.p.coord_offset_z, self.p.width, self.p.height, thickness=thickness
        )
        if not self.road_mask: raise RuntimeError("No road cells rasterized.")

        self._bfs_cache = {}

        # Depot & dispatcher
        self.depot = self._make(Depot, "ID_02")
        dep_param = getattr(self.p, "depot_pos", None)
        dep = snap_to_nearest(tuple(dep_param) if dep_param else (self.p.width//2, self.p.height//2), self.road_mask)
        self.grid.add_agents([self.depot], positions=[dep])

        self.dispatcher = self._make(Dispatcher, "ID_01")
        dsp_param = getattr(self.p, "dispatcher_pos", None)
        if dsp_param:
            dsp = snap_to_nearest(tuple(dsp_param), self.road_mask)
        else:
            nb = self.road_nbrs.get(dep, ()); dsp = nb[0] if nb else dep
        self.grid.add_agents([self.dispatcher], positions=[dsp])

        # Lights
        tls = []
        for i, t in enumerate(lights[:self.p.n_tlights], 1):
            tl = self._make(TrafficLight, f"ID_3{str(i).zfill(2)}")
            gx, gz = xform(t["x"], t["z"], self.p.coord_offset_x, self.p.coord_offset_z)
            pos = snap_to_nearest((gx, gz), self.road_mask)
            tls.append(tl); self.grid.add_agents([tl], positions=[pos])
        self.tlights = ap.AgentList(self, tls)

        # Bins
        bs = []
        random.shuffle(zones)
        for i, z in enumerate(zones[:self.p.n_bins], 1):
            b = self._make(TrashBin, f"ID_2{str(i).zfill(2)}")
            gx, gz = xform(int(z["center_x"]), int(z["center_z"]),
                           self.p.coord_offset_x, self.p.coord_offset_z)
            pos = snap_to_nearest((gx, gz), self.road_mask)
            bs.append(b); self.grid.add_agents([b], positions=[pos])
        self.bins = ap.AgentList(self, bs)
        for b in self.bins: b.notify_ready()

        # Index bins by position
        self.bins_by_pos = defaultdict(list)
        for b in self.bins: self.bins_by_pos[b.pos].append(b)

        # Trucks
        trs = []
        road_list = list(self.road_mask); random.shuffle(road_list)
        for i in range(self.p.n_trucks):
            tr = self._make(Truck, f"ID_1{str(i+1).zfill(2)}")
            start = road_list[i % len(road_list)]
            trs.append(tr); self.grid.add_agents([tr], positions=[start])
        self.trucks = ap.AgentList(self, trs)

        # Obstacles (spawn on free road cells)
        obs = []
        start_idx = len(trs)  # avoid overlapping first few road_list cells used by trucks
        used = {self.depot.pos} | {t.pos for t in self.trucks} | {b.pos for b in self.bins}
        j = 0
        for i in range(self.p.n_obstacles):
            o = self._make(RoadObstacle, f"ID_4{str(i+1).zfill(2)}")
            while j < len(road_list) and road_list[j] in used: j += 1
            pos = road_list[j % len(road_list)]
            used.add(pos); j += 1
            obs.append(o); self.grid.add_agents([o], positions=[pos])
        self.obstacles = ap.AgentList(self, obs)

        # KPIs, occupancy, history
        self.kpis_bins_done = 0
        self._occ = set()
        self.history = {'steps': []}
        self._start = time.perf_counter()

    def _make(self, cls, sid):
        class T(cls):
            def __init__(self, model):
                self.static_id = sid
                super().__init__(model)
        return T(self)

    # Occupancy — trucks, depot, and obstacles block movement
    def cell_free(self, pos):
        return (pos in self.road_mask) and (pos not in self._occ)

    def rebuild_occupancy(self):
        self._occ.clear()
        for a in list(self.trucks): self._occ.add(a.pos)
        for o in list(getattr(self, "obstacles", [])): self._occ.add(o.pos)
        self._occ.add(self.depot.pos)

    def blocked_by_red(self, cell):
        for tl in self.tlights:
            if tl.pos == cell and tl.phase == "R": return True
        return False

    def next_step_cached(self, src, dst):
        key = (src, dst)
        nxt = self._bfs_cache.get(key)
        if nxt is None:
            nxt = bfs_first_step(src, dst, self.road_nbrs); self._bfs_cache[key] = nxt
        return nxt

    def _record_history(self):
        step = {"tick": self.t, "agents": []}
        all_agents = [self.dispatcher, self.depot] + list(self.trucks) + list(self.bins) + list(self.tlights) + list(self.obstacles)
        for a in all_agents:
            ax, az = a.pos_xz()
            data = {"id": a.static_id, "type": a.kind, "x": ax, "z": az}
            if hasattr(a, "state"): data["state"] = a.state
            if a.kind == "truck":
                data.update({
                    "load": round(a.load, 2), "capacity": a.cap,
                    "collected_bins": a.collected_bins, "total_distance": a.total_distance,
                    "assigned_bin": a.assigned.static_id if a.assigned else None,
                    "last_action": a.last_action, "action_log": a.action_log[-10:],
                    "rotation_y": getattr(a, "spawn_angle", 0),
                    "corner_timer": 0, "stuck_counter": 0, "efficiency_score": 0,
                    "alpha": round(a.alpha, 3), "epsilon": round(a.eps, 3),
                    "last_reward": round(a.last_reward, 3),
                    "exploration_coverage": len(a.visited_cells), "is_in_corner": False
                })
            elif a.kind == "bin":
                data.update({"fill_level": round(a.fill, 2), "threshold": a.th, "ready_for_pickup": a.state == "Ready"})
            elif a.kind == "light":
                data.update({"phase": a.phase, "cycle_position": a.t})
            elif a.kind == "obstacle":
                data.update({"rotation_y": getattr(a, "spawn_angle", 0)})
            step["agents"].append(data)
        self.history["steps"].append(step)

    def step(self):
        self.t += 1
        self.rebuild_occupancy()

        self.tlights.step()
        self.bins.step()
        self.dispatcher.step()

        # Move obstacles then rebuild occupancy so trucks see them
        self.obstacles.step()
        self.rebuild_occupancy()

        self.trucks.step()

        if self.t % self.p.history_stride == 0 or self.t == 1:
            self._record_history()

    def end(self):
        os.makedirs("results/simulation_data", exist_ok=True)
        state = {"tick": self.t, "simulation_stats": {
                    "total_bins_done": self.kpis_bins_done,
                    "total_collected": int(sum(t.collected_bins for t in self.trucks)),
                    "total_distance": int(sum(t.total_distance for t in self.trucks)),
                    "open_tasks": len(self.dispatcher.open),
                    "pending_cfps": len(self.dispatcher.pending)}, "agents": []}

        def add_agent(a):
            x, z = a.pos_xz()
            data = {"id": a.static_id, "type": a.kind, "label": getattr(a, "kind", "entity"), "x": x, "z": z}
            if hasattr(a, "state"): data["state"] = a.state
            if a.kind == "truck":
                data.update({
                    "load": round(a.load, 2), "capacity": a.cap, "collected_bins": a.collected_bins,
                    "total_distance": a.total_distance, "assigned_bin": a.assigned.static_id if a.assigned else None,
                    "last_action": a.last_action, "action_log": a.action_log[-10:], "rotation_y": getattr(a, "spawn_angle", 0),
                    "corner_timer": 0, "stuck_counter": 0, "efficiency_score": 0,
                    "alpha": round(a.alpha, 3), "epsilon": round(a.eps, 3),
                    "last_reward": round(a.last_reward, 3), "exploration_coverage": len(a.visited_cells), "is_in_corner": False
                })
            elif a.kind == "bin":
                data.update({"fill_level": round(a.fill, 2), "threshold": a.th, "ready_for_pickup": a.state == "Ready"})
            elif a.kind == "light":
                data.update({"phase": a.phase, "cycle_position": a.t})
            elif a.kind == "obstacle":
                data.update({"rotation_y": getattr(a, "spawn_angle", 0)})
            state["agents"].append(data)

        add_agent(self.dispatcher); add_agent(self.depot)
        for grp in [self.trucks, self.bins, self.tlights, self.obstacles]:
            for a in grp: add_agent(a)

        with open("results/simulation_data/mas_final_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        with open("results/simulation_data/simulation_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        elapsed = time.perf_counter() - self._start
        print("\n=== Simulation finished ===")
        print(f"Ticks: {self.t}")
        print(f"Bins Done: {self.kpis_bins_done}")
        print(f"Fleet distance traveled: {int(sum(t.total_distance for t in self.trucks))}")
        print("Output JSON: results/simulation_data/mas_final_state.json")
        print("History JSON: results/simulation_data/simulation_history.json")

# -------------------------
# Params & run
# -------------------------
DEFAULT = {
    "base_path": "config_Sim",
    "width": 500, "height": 400,
    "coord_offset_x": 260, "coord_offset_z": 120,
    "steps": 5000,

    "n_trucks": 5, "n_bins": 40, "n_tlights": 10,
    "n_obstacles": 8,                     
    "obstacle_move_prob": 0.05,           
    "obstacle_obey_lights": True,         

    "depot_pos": (151, 299),
    "dispatcher_pos": (150, 274),

    "road_thickness": 2,

    "bin_init_min": 0.7, "bin_init_max": 1.4,
    "bin_threshold": 0.8, "bin_fill_rate": 0.05,
    "truck_capacity": 4.0,
    "pick_amount": 2.0,
    "unload_threshold": 0.6,

    "tl_cycle": 8, "tl_green": 6,

    # Energy system
    "energy_max": 120.0, "energy_per_move": 1.0,
    "charge_rate": 8.0, "energy_reserve": 10.0, "low_energy": 15.0,

    # Q-learning
    "q_alpha": 0.5, "q_gamma": 0.98, "q_epsilon": 0.08,

    # Dispatcher
    "cfp_timeout": 4,

    # History throttling
    "history_stride": 25,
}

if __name__ == "__main__":
    model = CityWasteV2(DEFAULT)
    res = model.run(steps=DEFAULT["steps"])
