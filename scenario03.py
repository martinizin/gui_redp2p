from flask import render_template, jsonify, request
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TOPOLOGY_DIR = os.getenv('TOPOLOGY_DIR', 'data')
EQPT_CONFIG_FILE = os.path.join(TOPOLOGY_DIR, 'eqpt_config.json')

def handle_scenario03():
    """Handles the logic for scenario 3."""
    maps_api_key = os.getenv('MAPS_API_KEY')
    return render_template('scenario3.html', maps_api_key=maps_api_key)

def get_topology_names():
    """Returns list of available topology files."""
    try:
        files = [f for f in os.listdir(TOPOLOGY_DIR) if f.endswith('.json') and f != 'eqpt_config.json']
        return jsonify(files)
    except FileNotFoundError:
        return jsonify([])

def get_topology_data(filename=None):
    """Returns topology data from specified file with enhanced parameter information."""
    if filename is None:
        filename = request.args.get('filename', 'CORONET_Global_Topology.json')
    
    filepath = os.path.join(TOPOLOGY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            topology_data = json.load(f)
        
        # Load equipment configuration
        eqpt_config = load_equipment_config()
        
        # Enhance topology data with parameter information
        enhanced_data = enhance_topology_with_params(topology_data, eqpt_config)
        
        return jsonify(enhanced_data)
    except FileNotFoundError:
        return jsonify({'error': f'File {filename} not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': f'Invalid JSON in file {filename}'}), 400

def load_equipment_config():
    """Load equipment configuration from eqpt_config.json."""
    try:
        with open(EQPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def enhance_topology_with_params(topology_data, eqpt_config):
    """Enhance topology data with parameter information from equipment config."""
    enhanced_data = topology_data.copy()
    
    # Add default parameters for different element types
    for element in enhanced_data.get('elements', []):
        element_type = element.get('type', '')
        
        if element_type == 'Transceiver':
            # Add default transceiver parameters
            element['parameters'] = get_transceiver_defaults()
            
        elif element_type == 'Fiber':
            # Fiber parameters are already in the topology, just ensure they exist
            if 'params' not in element:
                element['params'] = {}
            element['parameters'] = get_fiber_defaults(element.get('params', {}))
            
        elif element_type == 'Edfa':
            # Get EDFA parameters from equipment config
            type_variety = element.get('type_variety', 'std_medium_gain')
            edfa_config = find_edfa_config(eqpt_config, type_variety)
            operational = element.get('operational', {})
            element['parameters'] = get_edfa_defaults(edfa_config, operational)
    
    return enhanced_data

def get_transceiver_defaults():
    """Get default parameters for transceivers."""
    return {
        'p_rb': {'value': -20.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Received Signal Strength - Modify this value to adjust the signal strength'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'Transmission OSNR - Modify the OSNR value to optimize signal quality'},
        'sens': {'value': -25.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Receiver Sensitivity - The sensitivity level of the receiver to incoming signals'}
    }

def get_fiber_defaults(existing_params):
    """Get parameters for fiber elements."""
    return {
        'loss_coef': {'value': existing_params.get('loss_coef', 0.2), 'unit': 'dB/km', 'editable': False, 'tooltip': 'Fiber Loss Coefficient - The coefficient representing fiber loss rate'},
        'length_km': {'value': existing_params.get('length', 80), 'unit': 'km', 'editable': False, 'tooltip': 'Fiber Length (km) - The total length of the fiber section in kilometers'},
        'con_in': {'value': existing_params.get('con_in', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Input Connector - The type of connector used at the input of the fiber'},
        'con_out': {'value': existing_params.get('con_out', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Output Connector - The type of connector used at the output of the fiber'},
        'att_in': {'value': existing_params.get('att_in', 0.0), 'unit': 'dB', 'editable': False, 'tooltip': 'Input Losses - Losses encountered at the fiber input side'}
    }

def get_edfa_defaults(edfa_config, operational):
    """Get parameters for EDFA elements."""
    return {
        'gain_flatmax': {'value': edfa_config.get('gain_flatmax', 26), 'unit': 'dB', 'editable': True, 'tooltip': 'Maximum Flat Gain - The maximum gain achieved by the amplifier under flat conditions'},
        'gain_min': {'value': edfa_config.get('gain_min', 15), 'unit': 'dB', 'editable': True, 'tooltip': 'Minimum Gain - The minimum gain achievable by the amplifier'},
        'p_max': {'value': edfa_config.get('p_max', 23), 'unit': 'dBm', 'editable': True, 'tooltip': 'Maximum Power - The maximum output power provided by the amplifier'},
        'nf0': {'value': edfa_config.get('nf0', edfa_config.get('nf_min', 6)), 'unit': 'dB', 'editable': True, 'tooltip': 'Noise Factor - The noise figure of the amplifier affecting signal-to-noise ratio'},
        'gain_target': {'value': operational.get('gain_target', 20), 'unit': 'dB', 'editable': False, 'tooltip': 'Target Gain - The desired gain to be achieved by the amplifier based on operational settings'}
    }

def find_edfa_config(eqpt_config, type_variety):
    """Find EDFA configuration by type_variety."""
    edfa_configs = eqpt_config.get('Edfa', [])
    for config in edfa_configs:
        if config.get('type_variety') == type_variety:
            return config
    # Return default if not found
    return {'gain_flatmax': 26, 'gain_min': 15, 'p_max': 23, 'nf_min': 6}

def update_network_parameters():
    """Update network parameters for elements."""
    try:
        data = request.get_json()
        element_uid = data.get('element_uid')
        parameter_name = data.get('parameter_name')
        new_value = data.get('new_value')
        
        # Here you would typically save the updated parameters to a database
        # For now, we'll just return success
        return jsonify({
            'success': True,
            'message': f'Parameter {parameter_name} updated for element {element_uid}',
            'element_uid': element_uid,
            'parameter_name': parameter_name,
            'new_value': new_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

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

