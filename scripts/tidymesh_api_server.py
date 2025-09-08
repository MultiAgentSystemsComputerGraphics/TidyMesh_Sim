# api_server.py — TidyMesh Simulation API (Unity-ready)
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import time
from datetime import datetime
import threading
import logging

# ---------------------------------
# Logging / Flask
# ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TidyMeshAPI")

app = Flask(__name__)
CORS(app)  # Enable CORS for Unity WebGL builds

# ---------------------------------
# API config
# ---------------------------------
API_VERSION = "v3"
BASE_PATH = "/TidyMesh/Sim"
RESULTS_DIR = "../results/simulation_data"
VISUALIZATIONS_DIR = "../results/visualizations"

# ---------------------------------
# Simulator integration (import)
# ---------------------------------
try:
    # Import the simulator (same folder)
    from TidyMesh_Sim_v3 import run_simulation, DEFAULT as SIM_DEFAULT
except Exception as e:
    logger.warning(f"Simulator not importable yet: {e}")
    SIM_DEFAULT = {
        "n_trucks": 5, "n_bins": 40, "n_tlights": 10, "n_obstacles": 0, "steps": 1200
    }
    def run_simulation(params_overrides=None):
        raise RuntimeError("Simulator not available. Ensure TidyMesh_Sim_v3.py is importable.")

# ---------------------------------
# Cache
# ---------------------------------
cached_final_state = None
cached_simulation_history = None
last_modified_final = 0
last_modified_history = 0

# ---------------------------------
# Helpers
# ---------------------------------
def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RESULTS_DIR, filename)

def get_visualization_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, VISUALIZATIONS_DIR, filename)

def load_json_with_cache(filepath, cache_var, last_modified_var):
    """Load JSON with mtime cache."""
    global cached_final_state, cached_simulation_history, last_modified_final, last_modified_history
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None

        current_modified = os.path.getmtime(filepath)
        if cache_var is None or current_modified > last_modified_var:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if os.path.basename(filepath) == "mas_final_state.json":
                cached_final_state = data
                last_modified_final = current_modified
                return cached_final_state
            else:
                cached_simulation_history = data
                last_modified_history = current_modified
                return cached_simulation_history
        else:
            return cache_var
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def _safe_int(name, value, default, lo, hi):
    try:
        v = int(value)
        v = max(lo, min(hi, v))
        return v
    except Exception:
        logger.warning(f"[params] '{name}' invalid -> default {default}")
        return default

def _merge_overrides(body: dict) -> dict:
    """Pick the knobs Unity can set, with safe defaults and clamps."""
    d = SIM_DEFAULT.copy()
    body = body or {}

    d["n_trucks"]    = _safe_int("n_trucks",    body.get("n_trucks",    d.get("n_trucks", 5)),     5, 0, 50)
    d["n_bins"]      = _safe_int("n_bins",      body.get("n_bins",      d.get("n_bins", 40)),      40, 0, 1000)
    d["n_tlights"]   = _safe_int("n_tlights",   body.get("n_tlights",   d.get("n_tlights", 10)),   10, 0, 200)
    d["n_obstacles"] = _safe_int("n_obstacles", body.get("n_obstacles", d.get("n_obstacles", 0)),  0, 0, 200)
    d["steps"]       = _safe_int("steps",       body.get("steps",       d.get("steps", 1200)),     1200, 10, 2_000_000)

    # Keep any other SIM_DEFAULT keys if present
    for k, v in SIM_DEFAULT.items():
        d.setdefault(k, v)
    return d

# ---------------------------------
# Run control
# ---------------------------------
_sim_thread = None
_sim_status = {
    "state": "idle",          # idle | running | completed | error
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "params": None,
    "run_id": None,
}

# ---------------------------------
# Endpoints
# ---------------------------------
@app.route("/", methods=["GET"])
def api_info():
    return jsonify({
        "name": "TidyMesh Simulation API",
        "version": API_VERSION,
        "description": "REST API for Unity visualization of TidyMesh multi-agent waste collection simulation",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "final_state": f"{BASE_PATH}/{API_VERSION}/mas_final_state.json",
            "simulation_history": f"{BASE_PATH}/{API_VERSION}/simulation_history.json",
            "run": f"{BASE_PATH}/{API_VERSION}/run",
            "run_status": f"{BASE_PATH}/{API_VERSION}/run/status",
            "health": "/health",
            "status": "/status"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    final_state_path = get_file_path("mas_final_state.json")
    history_path = get_file_path("simulation_history.json")
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "files": {
            "final_state_exists": os.path.exists(final_state_path),
            "simulation_history_exists": os.path.exists(history_path),
            "final_state_size": os.path.getsize(final_state_path) if os.path.exists(final_state_path) else 0,
            "history_size": os.path.getsize(history_path) if os.path.exists(history_path) else 0
        }
    })

@app.route("/status", methods=["GET"])
def api_status():
    final_state_path = get_file_path("mas_final_state.json")
    history_path = get_file_path("simulation_history.json")
    status_data = {
        "api_version": API_VERSION,
        "server_time": datetime.now().isoformat(),
        "cache_status": {
            "final_state_cached": cached_final_state is not None,
            "history_cached": cached_simulation_history is not None,
            "last_reload_final": datetime.fromtimestamp(last_modified_final).isoformat() if last_modified_final > 0 else None,
            "last_reload_history": datetime.fromtimestamp(last_modified_history).isoformat() if last_modified_history > 0 else None
        },
        "file_info": {
            "final_state": {
                "exists": os.path.exists(final_state_path),
                "size_bytes": os.path.getsize(final_state_path) if os.path.exists(final_state_path) else 0,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(final_state_path)).isoformat() if os.path.exists(final_state_path) else None
            },
            "simulation_history": {
                "exists": os.path.exists(history_path),
                "size_bytes": os.path.getsize(history_path) if os.path.exists(history_path) else 0,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(history_path)).isoformat() if os.path.exists(history_path) else None
            }
        }
    }
    return jsonify(status_data)

# ---- Unity → start a simulation run ----
@app.route(f"{BASE_PATH}/{API_VERSION}/run", methods=["POST"])
def run_sim():
    global _sim_thread, _sim_status

    if _sim_status["state"] == "running":
        return jsonify({"status": "busy", "message": "A run is already in progress."}), 409

    body = request.get_json(silent=True) or {}
    params = _merge_overrides(body)

    def _worker(run_id, params):
        global _sim_status
        try:
            _sim_status.update({
                "state": "running",
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "last_error": None,
                "params": params,
                "run_id": run_id,
            })
            run_simulation(params_overrides=params)  # writes results/*.json
            _sim_status.update({
                "state": "completed",
                "finished_at": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.exception("Simulation failed")
            _sim_status.update({
                "state": "error",
                "finished_at": datetime.now().isoformat(),
                "last_error": str(e),
            })

    run_id = f"run_{int(time.time())}"
    _sim_thread = threading.Thread(target=_worker, args=(run_id, params), daemon=True)
    _sim_thread.start()

    return jsonify({
        "status": "started",
        "run_id": run_id,
        "params": {k: params[k] for k in ("n_trucks","n_bins","n_tlights","n_obstacles","steps")},
        "results": {
            "final_state": f"{BASE_PATH}/{API_VERSION}/mas_final_state.json",
            "history": f"{BASE_PATH}/{API_VERSION}/simulation_history.json"
        }
    }), 202

@app.route(f"{BASE_PATH}/{API_VERSION}/run/status", methods=["GET"])
def run_status():
    return jsonify(_sim_status)

# ---- JSON data endpoints (GET/POST) ----
@app.route(f"{BASE_PATH}/{API_VERSION}/mas_final_state.json", methods=["GET","POST"])
def get_final_state():
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "No data received"}), 400
            final_state_path = get_file_path("mas_final_state.json")
            os.makedirs(os.path.dirname(final_state_path), exist_ok=True)
            with open(final_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return jsonify({"status": "success", "message": "Final state data saved"}), 200
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        final_state_path = get_file_path("mas_final_state.json")
        data = load_json_with_cache(final_state_path, cached_final_state, last_modified_final)
        if data is None:
            return jsonify({"error": "Final state data not available",
                            "message": "Run the simulation first to generate data",
                            "timestamp": datetime.now().isoformat()}), 404
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "source": "TidyMesh Simulation",
            "version": API_VERSION,
            "data": data
        })

@app.route(f"{BASE_PATH}/{API_VERSION}/simulation_history.json", methods=["GET","POST"])
def get_simulation_history():
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "No data received"}), 400
            history_path = get_file_path("simulation_history.json")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return jsonify({"status": "success", "message": "Simulation history data saved"}), 200
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        history_path = get_file_path("simulation_history.json")
        data = load_json_with_cache(history_path, cached_simulation_history, last_modified_history)
        if data is None:
            return jsonify({"error": "Simulation history data not available",
                            "message": "Run the simulation first to generate data",
                            "timestamp": datetime.now().isoformat()}), 404
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "source": "TidyMesh Simulation",
            "version": API_VERSION,
            "data": data
        })

# ---- Static files ----
@app.route(f"{BASE_PATH}/{API_VERSION}/visualization/<filename>", methods=["GET"])
def get_visualization(filename):
    try:
        file_path = get_visualization_path(filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        return jsonify({"error": "Visualization file not found",
                        "filename": filename,
                        "timestamp": datetime.now().isoformat()}), 404
    except Exception as e:
        logger.error(f"Error serving visualization {filename}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route(f"{BASE_PATH}/{API_VERSION}/config/<filename>", methods=["GET"])
def get_config_file(filename):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../config_Sim", filename)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"timestamp": datetime.now().isoformat(),
                            "filename": filename,
                            "data": data})
        return jsonify({"error": "Configuration file not found",
                        "filename": filename,
                        "timestamp": datetime.now().isoformat()}), 404
    except Exception as e:
        logger.error(f"Error serving config {filename}: {e}")
        return jsonify({"error": str(e)}), 500

# ---- Errors ----
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Check the API documentation at the root endpoint",
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": [
            "/", "/health", "/status",
            f"{BASE_PATH}/{API_VERSION}/run",
            f"{BASE_PATH}/{API_VERSION}/run/status",
            f"{BASE_PATH}/{API_VERSION}/mas_final_state.json",
            f"{BASE_PATH}/{API_VERSION}/simulation_history.json",
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

# ---- Entrypoint ----
def start_server(host="localhost", port=5000, debug=False):
    logger.info(f"🚀 TidyMesh API → http://{host}:{port}")
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TidyMesh API Server for Unity Visualization")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, debug=args.debug)
