# tidymesh_api_server.py
# Flask API Server for TidyMesh Unity Visualization
# Provides REST endpoints for Unity to fetch simulation data
# Author: TidyMesh Development Team
# Date: August 31, 2025

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import time
from datetime import datetime
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Unity WebGL builds

# Configuration
API_VERSION = "v2"
BASE_PATH = "/TidyMesh/Sim"
RESULTS_DIR = "../results/simulation_data"
VISUALIZATIONS_DIR = "../results/visualizations"

# Global variables for caching
cached_final_state = None
cached_simulation_history = None
last_modified_final = 0
last_modified_history = 0

def get_file_path(filename):
    """Get absolute path for simulation data files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RESULTS_DIR, filename)

def get_visualization_path(filename):
    """Get absolute path for visualization files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, VISUALIZATIONS_DIR, filename)

def load_json_with_cache(filepath, cache_var, last_modified_var):
    """Load JSON file with caching based on modification time"""
    global cached_final_state, cached_simulation_history, last_modified_final, last_modified_history
    
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
            
        current_modified = os.path.getmtime(filepath)
        
        # Check if we need to reload the file
        if cache_var is None or current_modified > last_modified_var:
            logger.info(f"Loading/reloading file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update cache and modification time
            if 'final_state' in filepath:
                cached_final_state = data
                last_modified_final = current_modified
                return cached_final_state
            else:
                cached_simulation_history = data
                last_modified_history = current_modified
                return cached_simulation_history
        else:
            # Return cached data
            return cache_var
            
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        "name": "TidyMesh Simulation API",
        "version": API_VERSION,
        "description": "REST API for Unity visualization of TidyMesh multi-agent waste collection simulation",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "final_state": f"{BASE_PATH}/{API_VERSION}/mas_final_state.json",
            "simulation_history": f"{BASE_PATH}/{API_VERSION}/simulation_history.json",
            "health": "/health",
            "status": "/status"
        },
        "features": [
            "Real-time simulation data access",
            "Caching for improved performance", 
            "CORS enabled for Unity WebGL",
            "JSON format compatible with Unity",
            "Multi-layered Q-Learning data"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
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

@app.route('/status', methods=['GET'])
def api_status():
    """Detailed status endpoint"""
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

@app.route(f'{BASE_PATH}/{API_VERSION}/mas_final_state.json', methods=['GET', 'POST'])
def get_final_state():
    """
    GET: Return the final simulation state
    POST: Receive and store final simulation state from simulation
    """
    if request.method == 'POST':
        # Store data sent from simulation
        try:
            data = request.get_json()
            if data:
                final_state_path = get_file_path("mas_final_state.json")
                os.makedirs(os.path.dirname(final_state_path), exist_ok=True)
                
                with open(final_state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Final state data received and saved: {len(str(data))} characters")
                return jsonify({"status": "success", "message": "Final state data saved"}), 200
            else:
                return jsonify({"status": "error", "message": "No data received"}), 400
                
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    else:  # GET request
        final_state_path = get_file_path("mas_final_state.json")
        data = load_json_with_cache(final_state_path, cached_final_state, last_modified_final)
        
        if data is not None:
            # Add metadata for Unity
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "source": "TidyMesh Enhanced Multi-Layered Q-Learning Simulation",
                "version": API_VERSION,
                "data": data
            }
            return jsonify(response_data)
        else:
            return jsonify({
                "error": "Final state data not available",
                "message": "Run the simulation first to generate data",
                "timestamp": datetime.now().isoformat()
            }), 404

@app.route(f'{BASE_PATH}/{API_VERSION}/simulation_history.json', methods=['GET', 'POST'])
def get_simulation_history():
    """
    GET: Return the complete simulation history
    POST: Receive and store simulation history from simulation
    """
    if request.method == 'POST':
        # Store data sent from simulation
        try:
            data = request.get_json()
            if data:
                history_path = get_file_path("simulation_history.json")
                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Simulation history data received and saved: {len(str(data))} characters")
                return jsonify({"status": "success", "message": "Simulation history data saved"}), 200
            else:
                return jsonify({"status": "error", "message": "No data received"}), 400
                
        except Exception as e:
            logger.error(f"Error saving simulation history: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    else:  # GET request
        history_path = get_file_path("simulation_history.json")
        data = load_json_with_cache(history_path, cached_simulation_history, last_modified_history)
        
        if data is not None:
            # Add metadata for Unity
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "source": "TidyMesh Enhanced Multi-Layered Q-Learning Simulation",
                "version": API_VERSION,
                "data": data
            }
            return jsonify(response_data)
        else:
            return jsonify({
                "error": "Simulation history data not available", 
                "message": "Run the simulation first to generate data",
                "timestamp": datetime.now().isoformat()
            }), 404

@app.route(f'{BASE_PATH}/{API_VERSION}/visualization/<filename>', methods=['GET'])
def get_visualization(filename):
    """Serve visualization files (images, GIFs, etc.)"""
    try:
        file_path = get_visualization_path(filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({
                "error": "Visualization file not found",
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }), 404
    except Exception as e:
        logger.error(f"Error serving visualization {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route(f'{BASE_PATH}/{API_VERSION}/config/<filename>', methods=['GET'])
def get_config_file(filename):
    """Serve configuration files"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../config_Sim", filename)
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "data": data
            }
            return jsonify(response_data)
        else:
            return jsonify({
                "error": "Configuration file not found",
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }), 404
    except Exception as e:
        logger.error(f"Error serving config {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Custom 404 handler"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "Check the API documentation at the root endpoint",
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": [
            "/",
            "/health", 
            "/status",
            f"{BASE_PATH}/{API_VERSION}/mas_final_state.json",
            f"{BASE_PATH}/{API_VERSION}/simulation_history.json"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Custom 500 handler"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

def start_server(host='localhost', port=5000, debug=False):
    """Start the Flask server"""
    logger.info(f"🚀 Starting TidyMesh API Server...")
    logger.info(f"📡 Server will be available at: http://{host}:{port}")
    logger.info(f"🔗 API Endpoints:")
    logger.info(f"   📋 API Info: http://{host}:{port}/")
    logger.info(f"   💚 Health Check: http://{host}:{port}/health")
    logger.info(f"   📊 Status: http://{host}:{port}/status")
    logger.info(f"   🏁 Final State: http://{host}:{port}{BASE_PATH}/{API_VERSION}/mas_final_state.json")
    logger.info(f"   📜 History: http://{host}:{port}{BASE_PATH}/{API_VERSION}/simulation_history.json")
    logger.info(f"")
    logger.info(f"🎮 For Unity integration:")
    logger.info(f"   Use GET requests to fetch simulation data")
    logger.info(f"   Use POST requests to send new simulation data")
    logger.info(f"")
    logger.info(f"⚡ Enhanced Multi-Layered Q-Learning Data Available!")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TidyMesh API Server for Unity Visualization')
    parser.add_argument('--host', default='localhost', help='Host to bind the server (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, debug=args.debug)
