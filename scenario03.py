from flask import render_template, jsonify, request
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import networkx as nx
import sys
import traceback
import numpy as np

# Gnpy imports for route calculation
from gnpy.tools.json_io import load_equipment, load_network
from gnpy.core.utils import lin2db
from gnpy.core.elements import Transceiver, Roadm, Fiber, Edfa
from gnpy.tools.worker_utils import designed_network, transmission_simulation

# Cargar variables de entorno - specify explicit path for Docker compatibility
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Configuración
TOPOLOGY_DIR = os.getenv('TOPOLOGY_DIR', 'data')
EQPT_CONFIG_FILE = os.path.join(TOPOLOGY_DIR, 'eqpt_configv1.json')

# Cargar configuración de equipos (similar a scenario02.py)
edfa_equipment_data = {}
if os.path.exists(EQPT_CONFIG_FILE):
    with open(EQPT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        full_eqpt_config = json.load(f)
        if 'Edfa' in full_eqpt_config:
            for edfa_spec in full_eqpt_config['Edfa']:
                edfa_equipment_data[edfa_spec['type_variety']] = edfa_spec
else:
    print(f"Advertencia: Archivo de configuración de equipos no encontrado en {EQPT_CONFIG_FILE}")

def format_scientific_notation(value):
    """Format a number to match the exact notation used in the config file."""
    if value == 191.3e12:
        return '191.3e12'
    elif value == 32e9:
        return '32e9'
    elif value == 195.1e12:
        return '195.1e12'
    elif value == 50e9:
        return '50e9'
    else:
        # For other values, use a general scientific notation
        if value >= 1e12:
            return f"{value/1e12:.1f}e12"
        elif value >= 1e9:
            return f"{value/1e9:.0f}e9"
        else:
            return str(value)

def handle_scenario03():
    """Maneja la lógica para el escenario 3."""
    maps_api_key = os.getenv('MAPS_API_KEY')
    
    # Debug: Check if .env file exists and API key is loaded
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    print(f"DEBUG: .env file path: {env_file_path}")
    print(f"DEBUG: .env file exists: {os.path.exists(env_file_path)}")
    print(f"DEBUG: API key loaded: {maps_api_key is not None}")
    print(f"DEBUG: API key value: {maps_api_key[:10] if maps_api_key else 'None'}...")
    
    # Load equipment configuration to get SI parameters (matching notebook defaults)
    eqpt_config = load_equipment_config()
    si_config = {}
    
    # Extract SI configuration parameters
    if 'SI' in eqpt_config and len(eqpt_config['SI']) > 0:
        si_data = eqpt_config['SI'][0]  # Use first SI configuration
        si_config = {
            'f_min': format_scientific_notation(si_data.get('f_min', 191.3e12)),
            'baud_rate': format_scientific_notation(si_data.get('baud_rate', 32e9)),
            'f_max': format_scientific_notation(si_data.get('f_max', 196.1e12)),  # Matching notebook
            'spacing': format_scientific_notation(si_data.get('spacing', 50e9)),
            'power_dbm': si_data.get('power_dbm', 2),
            'tx_power_dbm': si_data.get('tx_power_dbm', 0),
            'roll_off': si_data.get('roll_off', 0.15),
            'tx_osnr': si_data.get('tx_osnr', 35),
            'sys_margins': si_data.get('sys_margins', 2)
        }
    else:
        # Fallback values matching notebook defaults exactly
        si_config = {
            'f_min': '191.3e12',
            'baud_rate': '32e9',
            'f_max': '196.1e12',  # Matching notebook
            'spacing': '50e9',
            'power_dbm': 2,
            'tx_power_dbm': 0,
            'roll_off': 0.15,
            'tx_osnr': 35,
            'sys_margins': 2
        }
    
    return render_template('scenario3.html', maps_api_key=maps_api_key, si_config=si_config)

def get_topology_names():
    """Devuelve lista de archivos de topología disponibles."""
    try:
        files = [f for f in os.listdir(TOPOLOGY_DIR) if f.endswith('.json') and f not in ['eqpt_configv1.json', 'eqpt_config_openroadm_ver4.json', 'eqpt_config.json']]
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
        
        # Validar requisitos de topología para redes bidireccionales
        validation_result = validate_topology_requirements(enhanced_data)
        enhanced_data['validation'] = validation_result
        
        return jsonify(enhanced_data)
    except FileNotFoundError:
        return jsonify({'error': f'Archivo {filename} no encontrado'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': f'JSON inválido en el archivo {filename}'}), 400

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
        'tx_osnr': {'value': 35.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - Modifique el valor OSNR para optimizar la calidad de señal'},
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

def validate_topology_requirements(topology_data):
    """
    Validar que la topología cumple con los requisitos para redes bidireccionales.
    Retorna un diccionario con resultado de validación y mensaje de error si aplica.
    Ahora usa validación permisiva que permite cálculos pero advierte sobre mejores prácticas.
    """
    try:
        elements = topology_data.get('elements', [])
        connections = topology_data.get('connections', [])
        
        # Contar elementos por tipo
        transceivers = [e for e in elements if e.get('type') == 'Transceiver']
        roadms = [e for e in elements if e.get('type') == 'Roadm']
        fibers = [e for e in elements if e.get('type') == 'Fiber']
        edfas = [e for e in elements if e.get('type') == 'Edfa']
        
        # Clasificar problemas por severidad
        critical_errors = []  # Bloquean cálculos
        warnings = []         # Permiten cálculos pero recomiendan mejoras
        
        # VALIDACIONES CRÍTICAS (requieren mínimo absoluto para gnpy)
        if len(transceivers) < 2:
            critical_errors.append(f"Se requieren al menos 2 Transceivers para calcular rutas, encontrados: {len(transceivers)}")
        
        if len(fibers) == 0 and len(connections) == 0:
            critical_errors.append("Se requiere al menos una conexión entre elementos para formar una red")
        
        # VALIDACIONES DE ADVERTENCIA (mejores prácticas)
        if len(transceivers) < 3:
            warnings.append(f"Recomendado: Al menos 3 Transceivers para redes robustas, encontrados: {len(transceivers)}")
        
        if len(roadms) < 2:
            warnings.append(f"Recomendado: Al menos 2 ROADMs para topologías complejas, encontrados: {len(roadms)}")
        
        if len(edfas) == 0:
            warnings.append("Recomendado: Usar amplificadores EDFA en enlaces largos para mejor rendimiento")
        
        # Validar conexiones bidireccionales (como advertencia, no bloqueo)
        bidirectional_warnings = validate_bidirectional_connections(connections, elements)
        if bidirectional_warnings:
            warnings.extend([f"Recomendado: {warning}" for warning in bidirectional_warnings])
        
        documentation_url = "https://guiatopologias.netlify.app/"
        
        # SI HAY ERRORES CRÍTICOS: bloquear cálculos
        if critical_errors:
            error_message = (
                "La topología no puede ser utilizada para cálculos debido a problemas críticos:\n\n" +
                "• " + "\n• ".join(critical_errors) + 
                f"\n\nConsulte la documentación completa en: {documentation_url}"
            )
            return {
                'valid': False,
                'errors': critical_errors,
                'warnings': warnings,
                'message': error_message,
                'documentation_url': documentation_url,
                'severity': 'critical'
            }
        
        # SI SOLO HAY ADVERTENCIAS: permitir cálculos con avisos
        if warnings:
            warning_message = (
                "La topología permite cálculos, pero se recomienda revisar las siguientes mejores prácticas:\n\n" +
                "• " + "\n• ".join(warnings) + 
                f"\n\nConsulte la documentación completa en: {documentation_url}"
            )
            return {
                'valid': True,
                'warnings': warnings,
                'message': warning_message,
                'documentation_url': documentation_url,
                'severity': 'warning'
            }
        
        # TODO BIEN
        return {
            'valid': True, 
            'message': 'La topología cumple con todos los requisitos y mejores prácticas',
            'severity': 'success'
        }
        
    except Exception as e:
        return {
            'valid': False, 
            'message': f'Error validando la topología: {str(e)}',
            'documentation_url': "https://guiatopologias.netlify.app/",
            'severity': 'error'
        }

def validate_bidirectional_connections(connections, elements):
    """
    Validar que las conexiones sean bidireccionales.
    Por cada conexión A -> B, debe existir una conexión B -> A.
    """
    errors = []
    
    # Crear mapeo de conexiones
    connection_pairs = {}
    element_uids = {e['uid'] for e in elements}
    
    for conn in connections:
        from_node = conn.get('from_node')
        to_node = conn.get('to_node')
        
        if not from_node or not to_node:
            continue
            
        # Verificar que ambos nodos existen en la topología
        if from_node not in element_uids or to_node not in element_uids:
            continue
            
        # Registrar la conexión
        connection_key = (from_node, to_node)
        reverse_key = (to_node, from_node)
        
        if connection_key not in connection_pairs:
            connection_pairs[connection_key] = False
        
        if reverse_key in connection_pairs:
            # Marcar ambas direcciones como encontradas
            connection_pairs[reverse_key] = True
            connection_pairs[connection_key] = True
        else:
            # Crear entrada para la conexión reversa
            connection_pairs[reverse_key] = False
    
    # Buscar conexiones sin dirección opuesta
    missing_connections = []
    for (from_node, to_node), has_reverse in connection_pairs.items():
        if not has_reverse:
            missing_connections.append(f"{from_node} ↔ {to_node}")
    
    if missing_connections:
        errors.append(
            f"Faltan conexiones bidireccionales para: {', '.join(missing_connections[:5])}" +
            (f" y {len(missing_connections) - 5} más" if len(missing_connections) > 5 else "")
        )
    
    return errors

def validate_topology():
    """Endpoint para validar requisitos de topología."""
    try:
        data = request.get_json()
        topology_filename = data.get('topology_filename')
        
        if not topology_filename:
            return jsonify({'error': 'Se requiere especificar el nombre del archivo de topología'}), 400
        
        filepath = os.path.join(TOPOLOGY_DIR, topology_filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Archivo de topología no encontrado: {topology_filename}'}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            topology_data = json.load(f)
        
        validation_result = validate_topology_requirements(topology_data)
        
        return jsonify({
            'success': True,
            'topology_filename': topology_filename,
            'validation': validation_result
        })
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Archivo de topología con formato JSON inválido'}), 400
    except Exception as e:
        return jsonify({'error': f'Error validando topología: {str(e)}'}), 500

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
            'message': f'Parámetro {parameter_name} actualizado para el elemento {element_uid}',
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
            
            # Validar la topología recién subida
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    topology_data = json.load(f)
                
                validation_result = validate_topology_requirements(topology_data)
                response_data = {
                    'message': 'Archivo subido exitosamente', 
                    'filename': filename,
                    'validation': validation_result
                }
                
                # Determinar mensaje según severidad
                if not validation_result['valid'] and validation_result.get('severity') == 'critical':
                    response_data['warning'] = 'El archivo se subió correctamente, pero la topología tiene problemas críticos que impiden los cálculos de rutas.'
                elif validation_result.get('warnings'):
                    response_data['info'] = 'El archivo se subió correctamente. La topología permite cálculos pero se recomienda revisar las mejores prácticas.'
                
                return jsonify(response_data), 200
                
            except (json.JSONDecodeError, Exception):
                # Si hay error validando, aún confirmar que el archivo se subió
                return jsonify({'message': 'Archivo subido exitosamente', 'filename': filename}), 200
        except Exception as e:
            return jsonify({'error': f'Error guardando el archivo: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Tipo de archivo no válido. Solo se permiten archivos .json'}), 400

def tipo(n):
    """Clasifica el tipo de nodo para mostrar en resultados."""
    if isinstance(n, Transceiver): return "TX/RX"
    if isinstance(n, Roadm): return "Roadm"
    if isinstance(n, Fiber): return "Fiber"
    if isinstance(n, Edfa): return "EDFA"
    return "Otro"

def calculate_routes():
    """Calcula las rutas óptimas entre dos nodos."""
    try:
        data = request.get_json()
        
        # Validar parámetros de entrada
        topology_filename = data.get('topology_filename')
        source_uid = data.get('source_node')
        destination_uid = data.get('destination_node')
        num_routes = data.get('number_of_routes', 1)
        calculation_criteria = data.get('calculation_criteria', 'osnr')
        
        # Parámetros del canal óptico
        optical_params = data.get('optical_parameters', {})
        
        if not all([topology_filename, source_uid, destination_uid]):
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        
        # Rutas de archivos
        EQPT = Path(EQPT_CONFIG_FILE)
        TOPO = Path(os.path.join(TOPOLOGY_DIR, topology_filename))
        
        if not EQPT.exists():
            return jsonify({'error': f'Archivo de configuración no encontrado: {EQPT}'}), 404
        
        if not TOPO.exists():
            return jsonify({'error': f'Archivo de topología no encontrado: {TOPO}'}), 404
        
        # Cargar red y configuración de equipos
        equipment = load_equipment(EQPT)
        network = load_network(TOPO, equipment)
        
        # Validar requisitos de topología antes de los cálculos
        with open(TOPO, 'r', encoding='utf-8') as f:
            topology_data = json.load(f)
        
        validation_result = validate_topology_requirements(topology_data)
        
        # Solo bloquear en errores críticos, permitir advertencias
        if not validation_result['valid'] and validation_result.get('severity') == 'critical':
            return jsonify({
                'error': 'Topología no válida para cálculos de rutas',
                'validation_message': validation_result['message'],
                'documentation_url': validation_result.get('documentation_url'),
                'validation_errors': validation_result.get('errors', []),
                'error_type': 'topology_validation'
            }), 400
        
        # Si hay advertencias pero la topología es válida, continuar con cálculos
        # Las advertencias se mostrarán en la respuesta final
        
        # Configurar parámetros del canal óptico (exactamente como en el notebook)
        si = list(equipment["SI"].values())[0]
        
        # Aplicar parámetros personalizados si se proporcionan
        if optical_params:
            si.f_min = float(optical_params.get('f_min', si.f_min))
            si.f_max = float(optical_params.get('f_max', si.f_max))
            si.spacing = float(optical_params.get('spacing', si.spacing))
            si.baud_rate = float(optical_params.get('baud_rate', si.baud_rate))
            si.roll_off = float(optical_params.get('roll_off', si.roll_off))
            si.tx_osnr = float(optical_params.get('tx_osnr', si.tx_osnr))
            si.sys_margins = float(optical_params.get('sys_margins', si.sys_margins))
            si.power_dbm = float(optical_params.get('power_dbm', si.power_dbm))
        
        # CRÍTICO: Aplicar power_range_db SIEMPRE (como en el notebook)
        si.power_range_db = [0, 0, 1]  # Valor fijo como en el notebook
        
        # Calcular número de canales (exactamente como en el notebook)
        num_channels = int(np.floor((si.f_max - si.f_min) / si.spacing)) + 1
        
        # Crear mapeo de UID a nodo
        uid2node = {n.uid: n for n in network.nodes()}
        
        # Obtener lista de transceivers para mostrar en resultados
        tx_nodes = [n for n in network.nodes() if isinstance(n, Transceiver)]
        tx_uids = [n.uid for n in tx_nodes]
        dst_uids = [uid for uid in tx_uids if uid != source_uid]
        
        # Validar que los nodos existen
        if source_uid not in uid2node:
            return jsonify({'error': f'Nodo origen {source_uid} no encontrado'}), 400
        if destination_uid not in uid2node:
            return jsonify({'error': f'Nodo destino {destination_uid} no encontrado'}), 400
        
        src, dst = uid2node[source_uid], uid2node[destination_uid]
        
        # Crear grafo para búsqueda de rutas
        G = nx.DiGraph()
        for u, v in network.edges():
            G.add_edge(u.uid, v.uid)
        
        # Buscar rutas simples
        try:
            paths_uid = list(nx.shortest_simple_paths(G, source_uid, destination_uid))[:num_routes]
        except nx.NetworkXNoPath:
            return jsonify({'error': f'No se encontraron rutas entre {source_uid} y {destination_uid}'}), 400
        
        # Evaluar rutas
        resultados = []
        for i, uid_path in enumerate(paths_uid):
            try:
                path_nodes = [uid2node[uid] for uid in uid_path]
                
                # Diseñar red y ejecutar simulación (exactamente como en el notebook)
                net_designed, req, ref_req = designed_network(
                    equipment, network,
                    source=src.uid,
                    destination=dst.uid,
                    nodes_list=[n.uid for n in path_nodes],
                    loose_list=['STRICT'],
                    args_power=si.power_dbm
                )
                
                path, _, _, infos = transmission_simulation(equipment, net_designed, req, ref_req)
                receiver = path[-1]
                
                # Copiar métricas desde infos hacia el receiver
                if hasattr(infos, "snr"):
                    receiver.snr = infos.snr
                if hasattr(infos, "osnr_ase"):
                    receiver.osnr_ase = infos.osnr_ase
                
                # Verificar que el receiver tiene las métricas necesarias
                if hasattr(receiver, "snr_01nm") and hasattr(receiver, "osnr_ase_01nm"):
                    # Calcular distancia total
                    dist_total = sum(getattr(n, 'length', getattr(getattr(n, 'params', None), 'length', 0))
                                   for n in path if isinstance(n, Fiber))
                    
                    # Crear información de la ruta
                    ruta_info = []
                    fibras_info = []
                    
                    for j, n in enumerate(path):
                        ruta_info.append({
                            'idx': j,
                            'uid': n.uid,
                            'tipo': tipo(n)
                        })
                        
                        if isinstance(n, Fiber):
                            length = getattr(n, 'length', getattr(n.params, 'length', 0))
                            fibras_info.append({
                                'uid': n.uid,
                                'length_km': length / 1000
                            })
                    
                    resultado = {
                        'ruta_num': i + 1,
                        'ruta_nodos': ruta_info,
                        'fibras': fibras_info,
                        'receptor_uid': receiver.uid,
                        'snr_01nm': float(receiver.snr_01nm.mean()),
                        'snr_bw': float(receiver.snr.mean()) if hasattr(receiver, "snr") else None,
                        'osnr_01nm': float(receiver.osnr_ase_01nm.mean()),
                        'osnr_bw': float(receiver.osnr_ase.mean()) if hasattr(receiver, "osnr_ase") else None,
                        'distancia_total_km': dist_total / 1000,
                        'uid_path': uid_path
                    }
                    
                    resultados.append(resultado)
                else:
                    print(f"⚠️  Ruta {i+1} falló: el receptor no tiene snr_01nm/osnr_ase_01nm.")
                    continue
                    
            except Exception as e:
                print(f"❌ Ruta {i+1} falló: {e}")
                traceback.print_exc()
                continue
        
        # Ordenar resultados según el criterio especificado
        if calculation_criteria == 'osnr':
            resultados.sort(key=lambda r: (-r['snr_01nm'], r['distancia_total_km']))
        else:  # 'distance'
            resultados.sort(key=lambda r: (r['distancia_total_km'], -r['snr_01nm']))
        
        # CRÍTICO: Renumerar rutas después de ordenar para que la mejor ruta sea siempre "Ruta 1"
        for i, resultado in enumerate(resultados):
            resultado['ruta_num'] = i + 1
        
        # Preparar respuesta
        response = {
            'success': True,
            'source_uid': source_uid,
            'destination_uid': destination_uid,
            'num_routes_requested': num_routes,
            'num_routes_found': len(resultados),
            'calculation_criteria': calculation_criteria,
            'transceivers_disponibles': tx_uids,
            'transceivers_destino_disponibles': dst_uids,
            'optical_parameters': {
                'f_min': si.f_min,
                'f_max': si.f_max,
                'spacing': si.spacing,
                'num_channels': num_channels,
                'baud_rate': si.baud_rate,
                'roll_off': si.roll_off,
                'tx_osnr': si.tx_osnr,
                'sys_margins': si.sys_margins,
                'power_dbm': si.power_dbm
            },
            'routes': resultados
        }
        
        # Incluir advertencias de validación si existen
        if validation_result.get('warnings'):
            response['validation_warnings'] = {
                'message': validation_result['message'],
                'warnings': validation_result['warnings'],
                'documentation_url': validation_result.get('documentation_url'),
                'severity': validation_result.get('severity', 'warning')
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error en calculate_routes: {e}")
        traceback.print_exc()
        
        # Determinar si es un error de validación de topología
        error_message = str(e)
        documentation_url = "https://guiatopologias.netlify.app/"
        
        if any(keyword in error_message.lower() for keyword in ['trx', 'fiber', 'roadm', 'edfa', 'connection', 'bidirectional']):
            return jsonify({
                'error': 'Error de topología en el cálculo de rutas',
                'details': f'Error interno del servidor: {str(e)}',
                'validation_message': f'La topología puede no cumplir con los requisitos para redes bidireccionales.\n\nError específico: {str(e)}\n\nConsulte la documentación para verificar los requisitos de topología.',
                'documentation_url': documentation_url,
                'error_type': 'topology_error'
            }), 500
        else:
            return jsonify({
                'error': 'Error interno del servidor',
                'details': str(e),
                'documentation_url': documentation_url,
                'error_type': 'server_error'
            }), 500
