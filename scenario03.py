from flask import render_template, jsonify, request
import json
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
TOPOLOGY_DIR = os.getenv('TOPOLOGY_DIR', 'data')
EQPT_CONFIG_FILE = os.path.join(TOPOLOGY_DIR, 'eqpt_config.json')

# Cargar configuración de equipos (similar a scenario02.py)
edfa_equipment_data = {}
if os.path.exists(EQPT_CONFIG_FILE):
    with open(EQPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        full_eqpt_config = json.load(f)
        if 'Edfa' in full_eqpt_config:
            for edfa_spec in full_eqpt_config['Edfa']:
                edfa_equipment_data[edfa_spec['type_variety']] = edfa_spec
else:
    print(f"Warning: Equipment configuration file not found at {EQPT_CONFIG_FILE}")

def handle_scenario03():
    """Maneja la lógica para el escenario 3."""
    maps_api_key = os.getenv('MAPS_API_KEY')
    return render_template('scenario3.html', maps_api_key=maps_api_key)

def get_topology_names():
    """Devuelve lista de archivos de topología disponibles."""
    try:
        files = [f for f in os.listdir(TOPOLOGY_DIR) if f.endswith('.json') and f != 'eqpt_config.json']
        return jsonify(files)
    except FileNotFoundError:
        return jsonify([])

def get_topology_data(filename=None):
    """Devuelve datos de topología del archivo especificado con información de parámetros mejorada."""
    if filename is None:
        filename = request.args.get('filename', 'CORONET_Global_Topology.json')
    
    filepath = os.path.join(TOPOLOGY_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            topology_data = json.load(f)
        
        # Cargar configuración de equipos
        eqpt_config = load_equipment_config()
        
        # Mejorar datos de topología con información de parámetros
        enhanced_data = enhance_topology_with_params(topology_data, eqpt_config)
        
        # Incluir configuración de equipos en la respuesta (como scenario02.py)
        enhanced_data['eqpt_config'] = eqpt_config
        
        return jsonify(enhanced_data)
    except FileNotFoundError:
        return jsonify({'error': f'File {filename} not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': f'Invalid JSON in file {filename}'}), 400

def load_equipment_config():
    """Cargar configuración de equipos desde eqpt_config.json."""
    try:
        with open(EQPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def enhance_topology_with_params(topology_data, eqpt_config):
    """Mejorar datos de topología con información de parámetros desde configuración de equipos."""
    enhanced_data = topology_data.copy()
    
    # Agregar parámetros por defecto para diferentes tipos de elementos
    for element in enhanced_data.get('elements', []):
        element_type = element.get('type', '')
        
        if element_type == 'Transceiver':
            # Agregar parámetros por defecto del transceptor
            element['parameters'] = get_transceiver_defaults()
            
        elif element_type == 'Fiber':
            # Los parámetros de fibra ya están en la topología, solo asegurar que existan
            if 'params' not in element:
                element['params'] = {}
            element['parameters'] = get_fiber_defaults(element.get('params', {}))
            
        elif element_type == 'Edfa':
            # Obtener parámetros EDFA desde configuración de equipos
            type_variety = element.get('type_variety', 'std_medium_gain')
            edfa_config = find_edfa_config(eqpt_config, type_variety)
            operational = element.get('operational', {})
            element['parameters'] = get_edfa_defaults(edfa_config, operational)
    
    return enhanced_data

def get_transceiver_defaults():
    """Obtener parámetros por defecto para transceptores."""
    return {
        'p_rb': {'value': -20.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Fuerza de Señal Recibida - Modifique este valor para ajustar la fuerza de la señal'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - Modifique el valor OSNR para optimizar la calidad de señal'},
        'sens': {'value': -25.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - El nivel de sensibilidad del receptor a las señales entrantes'}
    }

def get_fiber_defaults(existing_params):
    """Obtener parámetros para elementos de fibra."""
    return {
        'loss_coef': {'value': existing_params.get('loss_coef', 0.2), 'unit': 'dB/km', 'editable': False, 'tooltip': 'Coeficiente de Pérdida de Fibra - El coeficiente que representa la tasa de pérdida de la fibra'},
        'length_km': {'value': existing_params.get('length', 80), 'unit': 'km', 'editable': False, 'tooltip': 'Longitud de Fibra (km) - La longitud total de la sección de fibra en kilómetros'},
        'con_in': {'value': existing_params.get('con_in', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Entrada - El tipo de conector usado en la entrada de la fibra'},
        'con_out': {'value': existing_params.get('con_out', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Salida - El tipo de conector usado en la salida de la fibra'},
        'att_in': {'value': existing_params.get('att_in', 0.0), 'unit': 'dB', 'editable': False, 'tooltip': 'Pérdidas de Entrada - Pérdidas encontradas en el lado de entrada de la fibra'}
    }

def get_edfa_defaults(edfa_config, operational):
    """Obtener parámetros para elementos EDFA (usando lógica similar a scenario02.py)."""
    return {
        'gain_flatmax': {'value': edfa_config.get('gain_flatmax', 26), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Plana Máxima - La ganancia máxima alcanzada por el amplificador bajo condiciones planas'},
        'gain_min': {'value': edfa_config.get('gain_min', 15), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Mínima - La ganancia mínima alcanzable por el amplificador'},
        'p_max': {'value': edfa_config.get('p_max', 23), 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Máxima - La potencia de salida máxima proporcionada por el amplificador'},
        'nf0': {'value': edfa_config.get('nf0', edfa_config.get('nf_min', 6)), 'unit': 'dB', 'editable': True, 'tooltip': 'Factor de Ruido - La figura de ruido del amplificador que afecta la relación señal-ruido'},
        'gain_target': {'value': operational.get('gain_target', 20), 'unit': 'dB', 'editable': False, 'tooltip': 'Ganancia Objetivo - La ganancia deseada a ser alcanzada por el amplificador basada en configuraciones operacionales'}
    }

def find_edfa_config(eqpt_config, type_variety):
    """Encontrar configuración EDFA por type_variety (usando lógica similar a scenario02.py)."""
    # Primero intentar con el diccionario global cargado
    if type_variety in edfa_equipment_data:
        return edfa_equipment_data[type_variety]
    
    # Respaldo: buscar en eqpt_config pasado como parámetro
    edfa_configs = eqpt_config.get('Edfa', [])
    for config in edfa_configs:
        if config.get('type_variety') == type_variety:
            return config
    # Devolver valores por defecto si no se encuentra
    return {'gain_flatmax': 26, 'gain_min': 15, 'p_max': 23, 'nf_min': 6}

def update_network_parameters():
    """Actualizar parámetros de red para elementos."""
    try:
        data = request.get_json()
        element_uid = data.get('element_uid')
        parameter_name = data.get('parameter_name')
        new_value = data.get('new_value')
        
        # Aquí normalmente guardarías los parámetros actualizados en una base de datos
        # Por ahora, solo devolvemos éxito
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
    """Maneja la carga de archivos de topología."""
    if not file:
        return jsonify({'error': 'No hay archivo en la solicitud'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and file.filename.endswith('.json'):
        filename = file.filename
        filepath = os.path.join(TOPOLOGY_DIR, filename)
        
        # Asegurar que el directorio existe
        os.makedirs(TOPOLOGY_DIR, exist_ok=True)
        
        try:
            file.save(filepath)
            return jsonify({'message': 'Archivo subido exitosamente', 'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Tipo de archivo no válido. Solo se permiten archivos .json'}), 400

