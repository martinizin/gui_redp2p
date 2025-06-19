from flask import render_template, jsonify, request
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TOPOLOGY_DIR = os.getenv('TOPOLOGY_DIR', 'data')

def handle_scenario03():
    """Handles the logic for scenario 3."""
    maps_api_key = os.getenv('MAPS_API_KEY')
    return render_template('scenario3.html', maps_api_key=maps_api_key)

def get_topology_names():
    """Returns list of available topology files."""
    try:
        files = [f for f in os.listdir(TOPOLOGY_DIR) if f.endswith('.json')]
        return jsonify(files)
    except FileNotFoundError:
        return jsonify([])

def get_topology_data(filename=None):
    """Returns topology data from specified file."""
    if filename is None:
        filename = request.args.get('filename', 'CORONET_Global_Topology.json')
    
    filepath = os.path.join(TOPOLOGY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': f'File {filename} not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': f'Invalid JSON in file {filename}'}), 400

def upload_topology_file(file):
    """Handles topology file upload."""
    if not file:
        return jsonify({'error': 'No hay archivo en la solicitud'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and file.filename.endswith('.json'):
        filename = file.filename
        filepath = os.path.join(TOPOLOGY_DIR, filename)
        
        # Ensure directory exists
        os.makedirs(TOPOLOGY_DIR, exist_ok=True)
        
        try:
            file.save(filepath)
            return jsonify({'message': 'Archivo subido exitosamente', 'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Tipo de archivo no válido. Solo se permiten archivos .json'}), 400

