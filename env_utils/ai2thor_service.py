"""
AI2THOR Service - runs in Python 3.6 environment
Provides a REST API to interact with the AI2THOR environment
"""
import ai2thor.controller
import numpy as np
import json
from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser(description="AI2THOR Flask Service")
parser.add_argument('--x_display', type=str, default=0, help='X display to use, e.g., "0" or "1"')
parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask server on')
args = parser.parse_args()

# Initialize the AI2THOR controller
controller = ai2thor.controller.Controller()
controller.start(x_display=args.x_display)


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the AI2THOR environment with specific parameters"""
    data = request.json
    scene = data.get('scene', 'FloorPlan1')
    
    try:
        # Reset the scene
        controller.reset(scene)
        return jsonify({
            "success": True,
            "message": f"Environment initialized with scene {scene}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/interact', methods=['POST'])
def interact():
    """Perform an action in the AI2THOR environment"""
    data = request.json
    action = data.get('action')
    
    if not action:
        return jsonify({
            "success": False,
            "error": "Action parameter is required"
        }), 400
    
    try:
        # Extract action and parameters
        event = controller.step(action)

        image = Image.fromarray(np.uint8(event.frame), mode='RGB')
    
        # Save image to in-memory bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")  # or JPEG, depending on your needs
        buffer.seek(0)
        
        # Encode the bytes as base64 string
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        # Extract relevant information from the event
        response = {
            "success": True,
            "lastActionSuccess": event.metadata.get('lastActionSuccess', False),
            "lastAction": event.metadata.get('lastAction', False),
            "errorMessage": event.metadata.get('errorMessage', False),
            "agent": {
                "position": event.metadata.get('agent', {}).get('position'),
                "rotation": event.metadata.get('agent', {}).get('rotation'),
                "cameraHorizon": event.metadata.get('agent', {}).get('cameraHorizon')
            },
            "objects": event.metadata.get('objects', []),
            "visible_objects": [
                {
                    "objectId": obj.get('objectId'),
                    "objectType": obj.get('objectType'),
                    "position": obj.get('position')
                }
                for obj in event.metadata.get('objects', []) 
                if obj.get('visible', False)
            ],
            "inventoryObjects": event.metadata.get('inventoryObjects', []),
            "frame_bytes": img_str,
            "pose_discrete": event.pose_discrete,
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Check if the service is running"""
    return jsonify({"status": "AI2THOR service is running"})


print(f"Starting AI2THOR service on port {args.port}...")
app.run(host='0.0.0.0', port=args.port, debug=False)
