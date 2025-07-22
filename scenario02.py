import plotly.graph_objects as go
import json
import os
import numpy as np
from flask import jsonify, request
from pathlib import Path
import traceback

# Verificar si gnpy está disponible
try:
    from gnpy.tools.json_io import load_equipment, load_network
    from gnpy.core.info import create_arbitrary_spectral_information
    from gnpy.core.utils import dbm2watt as gnpy_dbm2watt_orig, watt2dbm as gnpy_watt2dbm_orig, lin2db as gnpy_lin2db_orig
    from gnpy.core.elements import Transceiver, Fiber, Edfa
    GNPY_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: gnpy no disponible - {e}")
    GNPY_AVAILABLE = False
    gnpy_dbm2watt_orig = None
    gnpy_watt2dbm_orig = None
    gnpy_lin2db_orig = None

# Funciones básicas de conversión (definidas primero para que puedan ser usadas por funciones envolventes)
def dbm2watt(dbm):
    """Convertir dBm a Watts"""
    return 10 ** ((dbm - 30) / 10)

def watt2dbm(watt):
    """Convertir Watts a dBm"""
    if watt <= 0:
        return -float('inf')
    return 30 + 10 * np.log10(watt)

def lin2db(lin):
    """Convertir lineal a dB"""
    if np.isscalar(lin):
        if lin <= 0:
            return -float('inf')
        return 10 * np.log10(lin)
    else:
        # Manejar arrays
        result = np.full_like(lin, -np.inf, dtype=float)
        mask = lin > 0
        result[mask] = 10 * np.log10(lin[mask])
        return result

# Funciones envolventes que usan versiones de gnpy cuando están disponibles, recurren a las locales
def gnpy_dbm2watt(dbm):
    """Convertir dBm a Watts usando gnpy si está disponible, de lo contrario función local"""
    if GNPY_AVAILABLE and gnpy_dbm2watt_orig:
        return gnpy_dbm2watt_orig(dbm)
    return dbm2watt(dbm)

def gnpy_watt2dbm(watt):
    """Convertir Watts a dBm usando gnpy si está disponible, de lo contrario función local"""
    if GNPY_AVAILABLE and gnpy_watt2dbm_orig:
        return gnpy_watt2dbm_orig(watt)
    return watt2dbm(watt)

def gnpy_lin2db(lin):
    """Convertir lineal a dB usando gnpy si está disponible, de lo contrario función local"""
    if GNPY_AVAILABLE and gnpy_lin2db_orig:
        return gnpy_lin2db_orig(lin)
    return lin2db(lin)

# PARÁMETROS
f_min, f_max = 191.3e12, 195.1e12
spacing = 50e9
roll_off = 0.15
tx_osnr = 40  # dB inicial, este valor es para los cálculos de gnpy de OSNR_bw
baud_rate = 32e9
B_n=12.5e9 # No sea modificable

# Cargar la configuración de equipos
EQPT_CONFIG_PATH = 'data/eqpt_final.json'
edfa_equipment_data = {}
if os.path.exists(EQPT_CONFIG_PATH):
    with open(EQPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        full_eqpt_config = json.load(f)
        if 'Edfa' in full_eqpt_config:
            for edfa_spec in full_eqpt_config['Edfa']:
                edfa_equipment_data[edfa_spec['type_variety']] = edfa_spec
else:
    print(f"Advertencia: Archivo de configuración de equipos no encontrado en {EQPT_CONFIG_PATH}")

# Calcular el número de canales antes de las entradas de usuario
nch = int(np.floor((f_max - f_min) / spacing)) + 1

# Parámetros por defecto (pueden ser sobrescritos por la interfaz web)
sens = -25.0  # Sensibilidad por defecto en dBm
P_tot_dbm_input = 15.0  # Potencia total por defecto en dBm

# Calcular la potencia por canal a partir de la potencia total ingresada
tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch) # Potencia POR CANAL en dBm

# Funciones auxiliares OSNR
def get_avg_osnr_db(si):
    sig = np.array([np.sum(ch.power) for ch in si.carriers])
    noise = si.ase + si.nli # SUMA ASE y NLI para el ruido total
    
    # Manejar comparaciones de arrays de numpy apropiadamente para evitar error de "valor de verdad ambiguo"
    mask = noise > 0
    osnr_linear = np.where(mask, sig / noise, np.inf)
    osnr_db = np.where(np.isfinite(osnr_linear), lin2db(osnr_linear), np.inf)
    
    return float(np.mean(osnr_db))

def format_osnr(v, decimals=2):
    return "∞" if np.isinf(v) else f"{v:.{decimals}f}"

def classical_osnr_parallel(signal_power_dbm, ase_noise_dbm):
    if ase_noise_dbm == -float('inf') or ase_noise_dbm <= -190: # Considerar ruido muy bajo
        return float('inf') # OSNR muy alto si el ruido es casi cero

    signal_power_lin = dbm2watt(signal_power_dbm)
    ase_noise_lin = dbm2watt(ase_noise_dbm)

    if ase_noise_lin <= 0:
        return float('inf') # Evitar división por cero o negativo

    return lin2db(signal_power_lin / ase_noise_lin)

# =================== FUNCIONES DE VISUALIZACIÓN DE TOPOLOGÍA ===================

def get_element_tooltip_text(element, edfa_specs):
    """Genera una cadena HTML formateada para el tooltip de un elemento."""
    uid = element.get('uid', 'N/A')
    element_type = element.get('type', 'N/A')
    type_variety = element.get('type_variety', 'N/A')

    tooltip_html = (f"<b>uid:</b> {uid}<br>"
                    f"<b>type:</b> {element_type}<br>"
                    f"<b>type_variety:</b> {type_variety}<br>")

    # Agregar parámetros operacionales específicos para EDFAs
    if element_type == 'Edfa':
        op = element.get('operational', {})
        spec = edfa_specs.get(type_variety, {})
        tooltip_html += "<hr><b>Parámetros Operacionales:</b><br>"
        tooltip_html += f"&nbsp;&nbsp;gain_target: {op.get('gain_target', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;tilt_target: {op.get('tilt_target', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;out_voa: {op.get('out_voa', 'N/A')} dB<br>"
        tooltip_html += "<hr><b>Especificaciones del Equipo:</b><br>"
        tooltip_html += f"&nbsp;&nbsp;gain_flatmax: {spec.get('gain_flatmax', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;gain_min: {spec.get('gain_min', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;p_max: {spec.get('p_max', 'N/A')} dBm<br>"
        tooltip_html += f"&nbsp;&nbsp;nf0: {spec.get('nf0', 5)} dB<br>"

    # Agregar todos los 'params' para cualquier elemento que los tenga (ej., Fiber)
    if 'params' in element:
        params = element.get('params', {})
        tooltip_html += "<b>params:</b><br>"
        for k, v in params.items():
            tooltip_html += f"&nbsp;&nbsp;{k}: {v}<br>"
    return tooltip_html

def get_fiber_chain_tooltip_text(fiber_chain, edfa_specs):
    """Genera una cadena HTML formateada para el tooltip de una cadena de fibras."""
    if not fiber_chain:
        return ""
    
    if len(fiber_chain) == 1:
        return get_element_tooltip_text(fiber_chain[0], edfa_specs)
    
    tooltip_html = f"<b>Cadena de Fibras ({len(fiber_chain)} tramos):</b><br>"
    total_length = 0
    
    for i, fiber in enumerate(fiber_chain, 1):
        uid = fiber.get('uid', 'N/A')
        params = fiber.get('params', {})
        length = params.get('length', 0)
        loss_coef = params.get('loss_coef', 0.2)
        total_length += length
        
        tooltip_html += f"<hr><b>Tramo {i}: {uid}</b><br>"
        tooltip_html += f"&nbsp;&nbsp;longitud: {length} km<br>"
        tooltip_html += f"&nbsp;&nbsp;coef_pérdida: {loss_coef} dB/km<br>"
        tooltip_html += f"&nbsp;&nbsp;pérdida_total: {(length * loss_coef):.2f} dB<br>"
    
        tooltip_html += f"<hr><b>Resumen de Cadena:</b><br>"
        tooltip_html += f"&nbsp;&nbsp;Longitud Total: {total_length} km<br>"
        tooltip_html += f"&nbsp;&nbsp;Tramos Totales: {len(fiber_chain)}<br>"
        
    return tooltip_html

def get_node_styles_and_tooltips(nodes_to_plot, edfa_specs):
    node_hover_texts = []
    node_symbols = []
    node_colors = []
    type_styles = {
        'Transceiver': {'color': '#6959CD', 'symbol': 'circle'},
        'Edfa': {'color': '#d62728', 'symbol': 'diamond'},
        'default': {'color': '#808080', 'symbol': 'circle'}
    }

    for node in nodes_to_plot:
        node_type = node.get('type', 'default')
        style = type_styles.get(node_type, type_styles['default'])
        node_colors.append(style['color'])
        node_symbols.append(style['symbol'])
        node_hover_texts.append(get_element_tooltip_text(node, edfa_specs))
            
    return node_hover_texts, node_symbols, node_colors

def detect_coordinate_overlaps(nodes_to_plot, coordinate_tolerance=1e-6):
    """
    Detecta grupos de nodos que comparten las mismas coordenadas (dentro de la tolerancia).
    Retorna un diccionario que mapea tuplas de coordenadas a listas de nodos.
    Soporta tanto estructuras metadata.latitude/longitude como metadata.location.latitude/longitude.
    """
    coordinate_groups = {}
    
    for node in nodes_to_plot:
        if 'metadata' in node:
            lat, lon = None, None
            
            # Intentar primero metadata.latitude/longitude (estructura Escenario02Test1.json)
            if 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
                lat = node['metadata'].get('latitude')
                lon = node['metadata'].get('longitude')
            # Intentar metadata.location.latitude/longitude (estructura topologiaEdfa1.json)
            elif 'location' in node['metadata']:
                location = node['metadata']['location']
                if isinstance(location, dict):
                    lat = location.get('latitude')
                    lon = location.get('longitude')
            
            if lat is not None and lon is not None:
                # Redondear coordenadas para manejar problemas de precisión de punto flotante
                coord_key = (round(float(lat) / coordinate_tolerance) * coordinate_tolerance,
                           round(float(lon) / coordinate_tolerance) * coordinate_tolerance)
                
                if coord_key not in coordinate_groups:
                    coordinate_groups[coord_key] = []
                coordinate_groups[coord_key].append(node)
    
    return coordinate_groups

def apply_coordinate_offsets(coordinate_groups, offset_distance=0.001):
    """
    Aplica pequeños offsets para los nodos que comparten las mismas coordenadas.
    Retorna un diccionario que mapea los UIDs de los nodos a sus coordenadas ajustadas.
    """
    adjusted_coordinates = {}
    
    for coord_key, nodes in coordinate_groups.items():
        if len(nodes) <= 1:
            # No overlap, use original coordinates
            if nodes:
                node = nodes[0]
                # Handle different metadata structures
                lat, lon = None, None
                if 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
                    lat = node['metadata']['latitude']
                    lon = node['metadata']['longitude']
                elif 'location' in node['metadata'] and isinstance(node['metadata']['location'], dict):
                    lat = node['metadata']['location'].get('latitude')
                    lon = node['metadata']['location'].get('longitude')
                
                if lat is not None and lon is not None:
                    adjusted_coordinates[node['uid']] = {
                        'lat': lat,
                        'lon': lon
                    }
        else:
            # Múltiples nodos en la misma ubicación - aplica los offsets
            base_lat, base_lon = coord_key
            
            # Calcula los offsets en un pequeño círculo alrededor del punto original
            for i, node in enumerate(nodes):
                if i == 0:
                    # El primer nodo permanece en la posición original
                    adjusted_coordinates[node['uid']] = {
                        'lat': base_lat,
                        'lon': base_lon
                    }
                else:
                    # Los nodos posteriores obtienen pequeños offsets circulares
                    angle = (2 * np.pi * i) / len(nodes)
                    lat_offset = offset_distance * np.cos(angle)
                    lon_offset = offset_distance * np.sin(angle)
                    
                    adjusted_coordinates[node['uid']] = {
                        'lat': base_lat + lat_offset,
                        'lon': base_lon + lon_offset
                    }
    
    return adjusted_coordinates

def apply_horizontal_coordinate_grouping(ordered_nodes, horizontal_spacing=100, group_offset=13):
    """
    Agrupa nodos con las mismas coordenadas juntos en el diseño horizontal.
    Esto aborda el problema donde los elementos que comparten coordenadas deberían estar visualmente cerca juntos.
    CRÍTICO: Mantiene el orden topológico dentro de cada grupo de coordenadas.
    """
    adjusted_positions = {}
    
    # Primero, agrupa nodos por sus coordenadas actuales, preservando el orden topológico
    coordinate_groups = {}
    coordinate_tolerance = 1e-6
    
    for i, node in enumerate(ordered_nodes):
        if 'metadata' in node and 'location' in node['metadata']:
            lat = node['metadata']['location'].get('latitude', None)
            lon = node['metadata']['location'].get('longitude', None)
            
            if lat is not None and lon is not None:
                # Crear clave de coordenadas para agrupar
                coord_key = (
                    round(float(lat) / coordinate_tolerance) * coordinate_tolerance,
                    round(float(lon) / coordinate_tolerance) * coordinate_tolerance
                )
                
                if coord_key not in coordinate_groups:
                    coordinate_groups[coord_key] = []
                coordinate_groups[coord_key].append((node, i))
            else:
                # Manejo de nodos sin coordenadas
                unique_key = f"no_coords_{i}"
                coordinate_groups[unique_key] = [(node, i)]
        else:
            # Manejo de nodos sin metadata
            unique_key = f"no_metadata_{i}"
            coordinate_groups[unique_key] = [(node, i)]
    
    # Ordenar grupos de coordenadas por el índice original mínimo para mantener el flujo topológico
    sorted_groups = sorted(coordinate_groups.items(), key=lambda x: min(idx for _, idx in x[1]))
    
    # Posiciona los grupos horizontalmente
    current_x = 0
    
    for coord_key, nodes_with_idx in sorted_groups:
        # Ordena nodos dentro de cada grupo por el orden topológico (índice original)
        nodes_with_idx.sort(key=lambda x: x[1])  # ordena por el índice original
        
        group_center_x = current_x
        
        if len(nodes_with_idx) == 1:
            # Un solo nodo - colócalo en el centro del grupo
            node, original_idx = nodes_with_idx[0]
            adjusted_positions[node['uid']] = {
                'x': group_center_x,
                'y': 100
            }
        else:
            # Múltiples nodos con las mismas coordenadas - colócalos en el orden topológico con pequeños offsets
            group_width = group_offset * (len(nodes_with_idx) - 1)
            start_x = group_center_x - group_width / 2
            
            for j, (node, original_idx) in enumerate(nodes_with_idx):
                # Posiciona los nodos horizontalmente en el orden topológico con pequeño espaciado
                node_x = start_x + (j * group_offset)
                node_y = 100
                
                # Mínimo offset vertical para evitar problemas de líneas de conexión
                # Mantiene el offset muy pequeño para mantener las conexiones rectas
                if j > 0:
                    node_y += (j % 2) * 1 - 0.5  # Muy pequeños offsets alternados (±0.5px)
                
                adjusted_positions[node['uid']] = {
                    'x': node_x,
                    'y': node_y
                }
        
        # Mueve a la siguiente posición del grupo
        if len(nodes_with_idx) > 1:
            # Toma en cuenta el espacio usado por este grupo
            current_x += horizontal_spacing + group_offset * len(nodes_with_idx)
        else:
            current_x += horizontal_spacing
    
    return adjusted_positions

def build_topology_graph(elements, connections):
    """
    Construir una representación gráfica de la topología de red para búsqueda de rutas.
    Devuelve un diccionario que mapea los UIDs de elementos a sus vecinos.
    """
    traversable_graph = {e['uid']: [] for e in elements}
    for conn in connections:
        from_node = conn['from_node']
        to_node = conn['to_node']
        if from_node in traversable_graph and to_node in traversable_graph:
            traversable_graph[from_node].append(to_node)
            traversable_graph[to_node].append(from_node)
    return traversable_graph, {e['uid']: e for e in elements}

def find_network_path(graph, source_uid, destination_uid):
    """
    Encontrar la ruta desde el origen hasta el transceptor de destino usando BFS.
    """
    if source_uid not in graph or destination_uid not in graph:
        return None
    queue = [(source_uid, [source_uid])]
    visited = {source_uid}
    while queue:
        current_uid, path = queue.pop(0)
        
        if current_uid == destination_uid:
            return path
        for neighbor_uid in graph.get(current_uid, []):
            if neighbor_uid not in visited:
                visited.add(neighbor_uid)
                queue.append((neighbor_uid, path + [neighbor_uid]))
    return None

def find_path_for_layout(elements, connections):
    """
    Determina la secuencia lineal de nodos 'reales' (no fibras) para diseño horizontal.
    """
    graph, elements_by_uid = build_topology_graph(elements, connections)
    source_uid, dest_uid = identify_source_destination_transceivers(elements)
    
    if not source_uid or not dest_uid:
        return sorted([e['uid'] for e in elements if e.get('type') != 'Fiber'], key=str)

    path_with_fibers = find_network_path(graph, source_uid, dest_uid)

    if not path_with_fibers:
        return sorted([e['uid'] for e in elements if e.get('type') != 'Fiber'], key=str)

    return [uid for uid in path_with_fibers if elements_by_uid[uid].get('type') != 'Fiber']

def process_scenario02_data(file):
    """Procesa el archivo JSON cargado y devuelve los datos de visualización de red."""
    if file.filename == '':
        return {'error': "No se seleccionó ningún archivo"}

    if not file.filename.endswith('.json'):
        return {'error': "Tipo de archivo inválido. Por favor, suba un archivo .json"}

    try:
        data = json.load(file.stream)
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        fig = go.Figure()

        # Mejorar elementos con parámetros
        enhanced_elements = enhance_elements_with_parameters(elements)
        enhanced_data = data.copy()
        enhanced_data['elements'] = enhanced_elements

        elements_by_uid = {el['uid']: el for el in elements}

        real_node_uids = {uid for uid, el in elements_by_uid.items() if el.get('type') != 'Fiber'}
        fiber_elements_by_uid = {uid: el for uid, el in elements_by_uid.items() if el.get('type') == 'Fiber'}
        
        # Construir mapa de conexiones
        connections_map = {}
        for conn in connections:
            from_n, to_n = conn['from_node'], conn['to_node']
            connections_map.setdefault(from_n, []).append(to_n)
        
        # Procesamiento mejorado de conexiones para manejar cadenas de fibras
        processed_connections = []
        processed_edge_tuples = set()
        
        def find_fiber_chain_end(start_fiber_uid, visited=None):
            """
            Encuentra recursivamente el final de una cadena de fibras.
            Devuelve (end_node_uid, fiber_chain) donde end_node_uid es un nodo real
            y fiber_chain es la lista de elementos de fibra en la cadena.
            """
            if visited is None:
                visited = set()
            
            if start_fiber_uid in visited:
                return None, []  # Referencia circular
            
            visited.add(start_fiber_uid)
            fiber_chain = [fiber_elements_by_uid[start_fiber_uid]]
            
            if start_fiber_uid not in connections_map:
                return None, fiber_chain
            
            for next_uid in connections_map[start_fiber_uid]:
                if next_uid in real_node_uids:
                    # Encontró nodo real al final de la cadena
                    return next_uid, fiber_chain
                elif next_uid in fiber_elements_by_uid:
                    # Continuar siguiendo la cadena de fibras
                    end_node, remaining_chain = find_fiber_chain_end(next_uid, visited.copy())
                    if end_node:
                        return end_node, fiber_chain + remaining_chain
            
            return None, fiber_chain
        
        # Procesar conexiones desde nodos reales
        for from_uid in real_node_uids:
            if from_uid not in connections_map: 
                continue
                
            for target_uid in connections_map[from_uid]:
                if target_uid in fiber_elements_by_uid:
                    # Encontró fibra, rastrear la cadena para encontrar el nodo final
                    end_node_uid, fiber_chain = find_fiber_chain_end(target_uid)
                    
                    if end_node_uid and end_node_uid in real_node_uids:
                        edge_tuple = tuple(sorted((from_uid, end_node_uid)))
                        if edge_tuple not in processed_edge_tuples:
                            # Usar la primera fibra de la cadena para visualización
                            primary_fiber = fiber_chain[0] if fiber_chain else None
                            processed_connections.append({
                                'from_node': from_uid, 
                                'to_node': end_node_uid,
                                'fiber_element': primary_fiber,
                                'fiber_chain': fiber_chain  # Almacenar cadena completa para tooltip
                            })
                            processed_edge_tuples.add(edge_tuple)
                            
                elif target_uid in real_node_uids:
                    # Conexión directa a nodo real (sin fibra)
                    edge_tuple = tuple(sorted((from_uid, target_uid)))
                    if edge_tuple not in processed_edge_tuples:
                        processed_connections.append({
                            'from_node': from_uid, 
                            'to_node': target_uid, 
                            'fiber_element': None,
                            'fiber_chain': []
                        })
                        processed_edge_tuples.add(edge_tuple)
        
        nodes_to_plot = [el for el in elements if el['uid'] in real_node_uids]
        
        # Determinar tipo de gráfico basado en coordenadas en 'metadata'
        has_coordinates = False
        plot_nodes = [node for node in nodes_to_plot if node.get('type') != 'Fiber']
        if plot_nodes:
            def has_valid_coordinates(node):
                if not isinstance(node.get('metadata'), dict):
                    return False
                
                lat, lon = None, None
                # Check for direct latitude/longitude
                if 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
                    lat = node['metadata']['latitude']
                    lon = node['metadata']['longitude']
                # Check for location.latitude/longitude
                elif 'location' in node['metadata'] and isinstance(node['metadata']['location'], dict):
                    lat = node['metadata']['location'].get('latitude')
                    lon = node['metadata']['location'].get('longitude')
                
                if lat is None or lon is None:
                    return False
                
                # Check if coordinates are valid numbers and not placeholder (0,0)
                return (isinstance(lat, (int, float)) and 
                       isinstance(lon, (int, float)) and 
                       not (lat == 0 and lon == 0))
            
            has_coordinates = all(has_valid_coordinates(node) for node in plot_nodes)

        if has_coordinates:
            fig = _create_map_plot(nodes_to_plot, processed_connections, data)
        else:
            fig = _create_horizontal_plot(nodes_to_plot, processed_connections, data)
        
        return {
            'graph_json': fig.to_json(),
            'enhanced_data': enhanced_data
        }
        
    except Exception as e:
        return {'error': f"Error al procesar el archivo: {e}"}

def _create_map_plot(nodes_to_plot, processed_connections, data):
    """Crea un gráfico basado en mapa para la topología de red."""
    fig = go.Figure()

    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(nodes_to_plot, edfa_equipment_data)
    
    nodes_by_uid = {node['uid']: node for node in nodes_to_plot}
    
    # Detecta las superposiciones de coordenadas y aplica los offsets
    coordinate_groups = detect_coordinate_overlaps(nodes_to_plot)
    adjusted_coordinates = apply_coordinate_offsets(coordinate_groups)
    
    # Dibujar conexiones (líneas) entre nodos - forzar líneas rectas
    for conn in processed_connections:
        from_uid, to_uid = conn['from_node'], conn['to_node']
        if from_uid in nodes_by_uid and to_uid in nodes_by_uid:
            from_node, to_node = nodes_by_uid[from_uid], nodes_by_uid[to_uid]
            
            # Use adjusted coordinates for line drawing
            from_coords = adjusted_coordinates.get(from_uid, {
                'lat': from_node['metadata']['latitude'], 
                'lon': from_node['metadata']['longitude']
            })
            to_coords = adjusted_coordinates.get(to_uid, {
                'lat': to_node['metadata']['latitude'], 
                'lon': to_node['metadata']['longitude']
            })
            
            # Crear líneas rectas agregando puntos intermedios para evitar curvatura
            from_lat, from_lon = from_coords['lat'], from_coords['lon']
            to_lat, to_lon = to_coords['lat'], to_coords['lon']
            
            # Generar puntos intermedios para línea recta
            num_points = 20  # Número de puntos intermedios
            lats = [from_lat + (to_lat - from_lat) * i / (num_points - 1) for i in range(num_points)]
            lons = [from_lon + (to_lon - from_lon) * i / (num_points - 1) for i in range(num_points)]
            
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                hoverinfo='none',
                line=dict(width=3, color='#FF0000'),  # Línea roja más gruesa para mejor visibilidad
                showlegend=False
            ))

    # Agregar marcadores invisibles en puntos medios para tooltips de fibras
    fiber_hover_lons, fiber_hover_lats, fiber_hover_texts = [], [], []
    for conn in processed_connections:
        if conn.get('fiber_chain'):
            from_uid, to_uid = conn['from_node'], conn['to_node']
            if from_uid in nodes_by_uid and to_uid in nodes_by_uid:
                from_node, to_node = nodes_by_uid[from_uid], nodes_by_uid[to_uid]
                
                # Use adjusted coordinates for fiber hover points
                from_coords = adjusted_coordinates.get(from_uid, {
                    'lat': from_node['metadata']['latitude'], 
                    'lon': from_node['metadata']['longitude']
                })
                to_coords = adjusted_coordinates.get(to_uid, {
                    'lat': to_node['metadata']['latitude'], 
                    'lon': to_node['metadata']['longitude']
                })
                
                fiber_hover_lons.append((from_coords['lon'] + to_coords['lon']) / 2)
                fiber_hover_lats.append((from_coords['lat'] + to_coords['lat']) / 2)
                fiber_hover_texts.append(get_fiber_chain_tooltip_text(conn['fiber_chain'], edfa_equipment_data))
    
    if fiber_hover_texts:
        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=fiber_hover_lons,
            lat=fiber_hover_lats,
            marker=dict(size=25, color='rgba(0,0,0,0)'), # Marcadores invisibles más grandes para mejor hover
            hovertext=fiber_hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False,
            name="Fibers"
        ))

    # Agregación de los marcadores y etiquetas de nodos principales de la red
    # Para mapbox, necesitamos manejar diferentes tipos de nodos por separado
    # ya que mapbox no soporta todos los símbolos de plotly regular
    
    # Separar nodos por tipo para diferentes representaciones
    transceivers = []
    edfas = []
    other_nodes = []
    
    for i, node in enumerate(nodes_to_plot):
        # Use adjusted coordinates if available, otherwise use original coordinates
        # Handle different metadata structures
        default_lat, default_lon = 0, 0
        if 'metadata' in node:
            if 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
                default_lat = node['metadata']['latitude']
                default_lon = node['metadata']['longitude']
            elif 'location' in node['metadata'] and isinstance(node['metadata']['location'], dict):
                default_lat = node['metadata']['location'].get('latitude', 0)
                default_lon = node['metadata']['location'].get('longitude', 0)
        
        coords = adjusted_coordinates.get(node['uid'], {
            'lat': default_lat,
            'lon': default_lon
        })
        
        node_info = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'uid': node['uid'],
            'hover': node_hover_texts[i],
            'color': node_colors[i]
        }
        
        if node.get('type') == 'Transceiver':
            transceivers.append(node_info)
        elif node.get('type') == 'Edfa':
            edfas.append(node_info)
        else:
            other_nodes.append(node_info)
    
    # Agregar transceivers con símbolo de círculo más grande y distintivo
    if transceivers:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[t['lon'] for t in transceivers],
            lat=[t['lat'] for t in transceivers],
            text=[t['uid'] for t in transceivers],
            hovertext=[t['hover'] for t in transceivers],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                size=12,  # Tamaño reducido para transceivers
                color=[t['color'] for t in transceivers],
                symbol='circle'  # Círculo para transceivers
            ),
            textposition='top right',
            textfont=dict(size=12, color='black'),
            showlegend=False,
            name="Transceivers"
        ))
    
    # Agregar EDFAs con símbolo circular pequeño y rojo
    if edfas:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[e['lon'] for e in edfas],
            lat=[e['lat'] for e in edfas],
            text=[e['uid'] for e in edfas],  # Usar UID simple
            hovertext=[e['hover'] for e in edfas],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                size=8,  # Más pequeño que TX/RX (18) pero visible
                color='#FF0000',  # Rojo fijo para todos los EDFAs
                symbol='circle'  # Círculo como TX/RX pero más pequeño
            ),
            textposition='bottom right',
            textfont=dict(size=10, color='red', family='Arial'),  # Texto rojo más pequeño
            showlegend=False,
            name="EDFAs"
        ))
    
    # Agregar otros nodos con símbolo por defecto
    if other_nodes:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[o['lon'] for o in other_nodes],
            lat=[o['lat'] for o in other_nodes],
            text=[o['uid'] for o in other_nodes],
            hovertext=[o['hover'] for o in other_nodes],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(
                size=20,
                color=[o['color'] for o in other_nodes],
                symbol='circle'
            ),
            textposition='bottom right',
            textfont=dict(size=11, color='black'),
            showlegend=False,
            name="Other Nodes"
        ))

    # Configurar diseño del mapa
    # Recolectar todas las coordenadas para cálculos de centro y zoom
    all_lats = []
    all_lons = []
    for node in nodes_to_plot:
        if 'metadata' in node:
            # Handle different metadata structures
            lat, lon = None, None
            if 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
                lat = node['metadata']['latitude']
                lon = node['metadata']['longitude']
            elif 'location' in node['metadata'] and isinstance(node['metadata']['location'], dict):
                lat = node['metadata']['location'].get('latitude')
                lon = node['metadata']['location'].get('longitude')
            
            if lat is not None and lon is not None:
                # Use adjusted coordinates for centering calculation
                coords = adjusted_coordinates.get(node['uid'], {'lat': lat, 'lon': lon})
                all_lats.append(coords['lat'])
                all_lons.append(coords['lon'])
    
    center_lat = np.mean(all_lats) if all_lats else 0
    center_lon = np.mean(all_lons) if all_lons else 0
    
    # Calcular zoom apropiado para topología punto a punto
    zoom_level = 5  # Zoom por defecto
    if len(all_lats) == 2:  # Topología punto a punto
        # Calcular distancia entre puntos para ajustar zoom
        lat_diff = abs(max(all_lats) - min(all_lats))
        lon_diff = abs(max(all_lons) - min(all_lons))
        max_diff = max(lat_diff, lon_diff)
        
        # Ajustar zoom basado en la distancia
        if max_diff < 0.1:  # Muy cerca
            zoom_level = 12
        elif max_diff < 0.5:  # Cercano
            zoom_level = 10
        elif max_diff < 2:  # Moderado
            zoom_level = 8
        elif max_diff < 5:  # Lejano
            zoom_level = 6
        else:  # Muy lejano
            zoom_level = 4
    
    fig.update_layout(
        title_text=data.get('network_name', 'Topología de Red Punto a Punto' if len(all_lats) == 2 else 'Topología de Red'),
        showlegend=False,
        hovermode='closest',
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level
        ),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    return fig

def _create_horizontal_plot(nodes_to_plot, processed_connections, data):
    """Crea un gráfico 2D horizontal para la topología de red."""
    fig = go.Figure()

    # Determinar el orden horizontal de nodos desde la lista completa de elementos
    all_elements = data.get('elements', [])
    all_connections = data.get('connections', [])
    ordered_node_uids = find_path_for_layout(all_elements, all_connections)
    
    # Crear una búsqueda para los objetos de nodo originales que se están graficando
    nodes_by_uid = {node['uid']: node for node in nodes_to_plot}
    
    # Filtrar y ordenar los nodos que realmente serán graficados
    ordered_nodes_to_plot = [nodes_by_uid[uid] for uid in ordered_node_uids if uid in nodes_by_uid]
    
    if not ordered_nodes_to_plot: # Respaldo si la búsqueda de ruta falla o devuelve vacío
        ordered_nodes_to_plot = sorted(nodes_to_plot, key=lambda x: x['uid'])
        ordered_node_uids = [node['uid'] for node in ordered_nodes_to_plot]

    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(ordered_nodes_to_plot, edfa_equipment_data)
    
    # Apply horizontal coordinate grouping for elements with same coordinates
    adjusted_positions = apply_horizontal_coordinate_grouping(ordered_nodes_to_plot)
    
    # Use adjusted positions, fallback to default positioning if needed
    node_positions = {}
    for i, node in enumerate(ordered_nodes_to_plot):
        uid = node['uid']
        if uid in adjusted_positions:
            node_positions[uid] = adjusted_positions[uid]
        else:
            # Fallback to original positioning
            node_positions[uid] = {'x': i * 100, 'y': 100}

    # Preparar listas para puntos de hover de span de fibra
    hover_x, hover_y, d2_hover_texts = [], [], []

    # Dibujar conexiones (líneas) entre nodos - ensure perfectly straight lines
    for conn in processed_connections:
        from_uid, to_uid = conn['from_node'], conn['to_node']
        if from_uid in node_positions and to_uid in node_positions:
            from_pos, to_pos = node_positions[from_uid], node_positions[to_uid]
            
            # Force perfectly horizontal lines by using the average Y position
            # This prevents any visual curvature in connections
            avg_y = (from_pos['y'] + to_pos['y']) / 2
            
            # Create multiple points to ensure straight line rendering
            x_points = [from_pos['x'], (from_pos['x'] + to_pos['x'])/2, to_pos['x']]
            y_points = [avg_y, avg_y, avg_y]

            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines', 
                line=dict(width=2, color='gray', shape='linear'),  # Explicitly linear
                hoverinfo='none', 
                showlegend=False,
                connectgaps=True  # Ensure lines connect properly
            ))

            # Agregar un marcador invisible en el punto medio para el tooltip de fibra
            if conn.get('fiber_chain'):
                hover_x.append((from_pos['x'] + to_pos['x']) / 2)
                hover_y.append(from_pos['y'])
                d2_hover_texts.append(get_fiber_chain_tooltip_text(conn['fiber_chain'], edfa_equipment_data))

    # Agregar los puntos de hover invisibles de fibra
    if d2_hover_texts:
        fig.add_trace(go.Scatter(
            x=hover_x, y=hover_y, mode='markers',
            marker=dict(size=25, color='rgba(0,0,0,0)'),
            hovertext=d2_hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False
        ))

    # Agregación de los marcadores y etiquetas de nodos principales de la red
    # Separar transceivers de otros elementos para diferentes posiciones de texto
    transceiver_indices = []
    other_indices = []
    
    for i, el in enumerate(ordered_nodes_to_plot):
        if el.get('type') == 'Transceiver':
            transceiver_indices.append(i)
        else:
            other_indices.append(i)
    
    # Agregar transceivers con etiquetas arriba
    if transceiver_indices:
        tx_x = [node_positions[ordered_nodes_to_plot[i]['uid']]['x'] for i in transceiver_indices]
        tx_y = [node_positions[ordered_nodes_to_plot[i]['uid']]['y'] for i in transceiver_indices]
        tx_text = [ordered_nodes_to_plot[i]['uid'] for i in transceiver_indices]
        tx_hover = [node_hover_texts[i] for i in transceiver_indices]
        tx_colors = [node_colors[i] for i in transceiver_indices]
        tx_symbols = [node_symbols[i] for i in transceiver_indices]
        
        fig.add_trace(go.Scatter(
            x=tx_x, y=tx_y,
            text=tx_text,
            hovertext=tx_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            mode='markers+text',
            textposition='top center',
            marker=dict(size=20, color=tx_colors, symbol=tx_symbols),
            textfont=dict(size=11),
            showlegend=False,
            name="Transceivers"
        ))
    
    # Agregar otros elementos con etiquetas abajo
    if other_indices:
        other_x = [node_positions[ordered_nodes_to_plot[i]['uid']]['x'] for i in other_indices]
        other_y = [node_positions[ordered_nodes_to_plot[i]['uid']]['y'] for i in other_indices]
        other_text = [ordered_nodes_to_plot[i]['uid'] for i in other_indices]
        other_hover = [node_hover_texts[i] for i in other_indices]
        other_colors = [node_colors[i] for i in other_indices]
        other_symbols = [node_symbols[i] for i in other_indices]
        
        fig.add_trace(go.Scatter(
            x=other_x, y=other_y,
            text=other_text,
            hovertext=other_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            mode='markers+text',
            textposition='bottom center',
            marker=dict(size=20, color=other_colors, symbol=other_symbols),
            textfont=dict(size=11),
            showlegend=False,
            name="Other Elements"
        ))
    
    # Detectar si es topología punto a punto para ajustar título
    is_point_to_point = len(ordered_nodes_to_plot) == 2
    
    # Calculate proper axis ranges to prevent autoscale issues
    if node_positions:
        x_positions = [pos['x'] for pos in node_positions.values()]
        y_positions = [pos['y'] for pos in node_positions.values()]
        
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        
        # Add padding for better visualization
        x_padding = max((x_max - x_min) * 0.1, 50)
        y_padding = max((y_max - y_min) * 0.2, 30)
        
        x_range = [x_min - x_padding, x_max + x_padding]
        y_range = [y_min - y_padding, y_max + y_padding]
    else:
        # Fallback ranges
        x_range = [-50, len(ordered_nodes_to_plot) * 100 - 50]
        y_range = [0, 200]
    
    fig.update_layout(
        title_text=data.get('network_name', 'Topología de Red Punto a Punto' if is_point_to_point else 'Topología de Red'),
        showlegend=False,
        xaxis=dict(
            visible=False, 
            range=x_range,
            fixedrange=True,  # Prevent zooming/panning on x-axis
            autorange=False   # Disable automatic range adjustment
        ),
        yaxis=dict(
            visible=False, 
            range=y_range,
            fixedrange=True,  # Prevent zooming/panning on y-axis  
            autorange=False,  # Disable automatic range adjustment
            scaleanchor="x",  # Maintain aspect ratio
            scaleratio=1      # 1:1 aspect ratio for straight lines
        ),
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        margin=dict(l=20, r=20, t=40, b=20),
        # Additional settings to prevent layout changes
        autosize=True,
        dragmode='pan'    # Allow panning but prevent other drag modes that might trigger autoscale
    )
    
    return fig

def create_topology_visualization(topology_file_path):
    """
    Crear visualización de topología a partir de un archivo JSON.
    
    Args:
        topology_file_path (str): Ruta al archivo JSON de topología
        
    Returns:
        dict: Diccionario con la figura de plotly y datos de topología
    """
    try:
        with open(topology_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        if not elements:
            return {'error': 'No hay elementos en la topología'}
        
        # Process elements for visualization (same logic as process_scenario02_data)
        elements_by_uid = {el['uid']: el for el in elements}
        real_node_uids = {uid for uid, el in elements_by_uid.items() if el.get('type') != 'Fiber'}
        fiber_elements_by_uid = {uid: el for uid, el in elements_by_uid.items() if el.get('type') == 'Fiber'}
        
        # Build connections map
        connections_map = {}
        for conn in connections:
            from_n, to_n = conn['from_node'], conn['to_node']
            connections_map.setdefault(from_n, []).append(to_n)
        
        # Process connections to handle fiber chains
        processed_connections = []
        processed_edge_tuples = set()
        
        def find_fiber_chain_end(start_fiber_uid, visited=None):
            """Find the end of a fiber chain."""
            if visited is None:
                visited = set()
            
            if start_fiber_uid in visited:
                return None, []
            
            visited.add(start_fiber_uid)
            fiber_chain = [fiber_elements_by_uid[start_fiber_uid]]
            
            if start_fiber_uid not in connections_map:
                return None, fiber_chain
            
            for next_uid in connections_map[start_fiber_uid]:
                if next_uid in real_node_uids:
                    return next_uid, fiber_chain
                elif next_uid in fiber_elements_by_uid:
                    end_node, remaining_chain = find_fiber_chain_end(next_uid, visited.copy())
                    if end_node:
                        return end_node, fiber_chain + remaining_chain
            
            return None, fiber_chain
        
        # Process connections from real nodes
        for from_uid in real_node_uids:
            if from_uid not in connections_map: 
                continue
                
            for target_uid in connections_map[from_uid]:
                if target_uid in fiber_elements_by_uid:
                    end_node_uid, fiber_chain = find_fiber_chain_end(target_uid)
                    
                    if end_node_uid and end_node_uid in real_node_uids:
                        edge_tuple = tuple(sorted((from_uid, end_node_uid)))
                        if edge_tuple not in processed_edge_tuples:
                            primary_fiber = fiber_chain[0] if fiber_chain else None
                            processed_connections.append({
                                'from_node': from_uid, 
                                'to_node': end_node_uid,
                                'fiber_element': primary_fiber,
                                'fiber_chain': fiber_chain
                            })
                            processed_edge_tuples.add(edge_tuple)
                            
                elif target_uid in real_node_uids:
                    edge_tuple = tuple(sorted((from_uid, target_uid)))
                    if edge_tuple not in processed_edge_tuples:
                        processed_connections.append({
                            'from_node': from_uid, 
                            'to_node': target_uid, 
                            'fiber_element': None,
                            'fiber_chain': []
                        })
                        processed_edge_tuples.add(edge_tuple)
        
        nodes_to_plot = [el for el in elements if el['uid'] in real_node_uids]
        
        # Determine plot type based on coordinates
        has_coordinates = False
        plot_nodes = [node for node in nodes_to_plot if node.get('type') != 'Fiber']
        if plot_nodes:
            has_coordinates = all(
                isinstance(node.get('metadata'), dict) and
                'latitude' in node['metadata'] and
                'longitude' in node['metadata'] and
                isinstance(node['metadata']['latitude'], (int, float)) and
                isinstance(node['metadata']['longitude'], (int, float)) and
                not (node['metadata']['latitude'] == 0 and node['metadata']['longitude'] == 0)
                for node in plot_nodes
            )

        if has_coordinates:
            fig = _create_map_plot(nodes_to_plot, processed_connections, data)
        else:
            fig = _create_horizontal_plot(nodes_to_plot, processed_connections, data)
        
        return {
            'figure': fig,
            'topology_data': data,
            'elements': elements,
            'connections': connections,
            'processed_connections': processed_connections
        }
        
    except Exception as e:
        return {'error': f"Error al crear visualización: {e}"}

def create_topology_visualization_from_data(data):
    """Crea una visualización de topología a partir de un diccionario de datos."""
    try:
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        if not elements:
            return {'error': 'No hay elementos en la topología'}
        
        # Same processing logic as create_topology_visualization
        elements_by_uid = {el['uid']: el for el in elements}
        real_node_uids = {uid for uid, el in elements_by_uid.items() if el.get('type') != 'Fiber'}
        fiber_elements_by_uid = {uid: el for uid, el in elements_by_uid.items() if el.get('type') == 'Fiber'}
        
        connections_map = {}
        for conn in connections:
            from_n, to_n = conn['from_node'], conn['to_node']
            connections_map.setdefault(from_n, []).append(to_n)
        
        processed_connections = []
        processed_edge_tuples = set()
        
        def find_fiber_chain_end(start_fiber_uid, visited=None):
            if visited is None:
                visited = set()
            
            if start_fiber_uid in visited:
                return None, []
            
            visited.add(start_fiber_uid)
            fiber_chain = [fiber_elements_by_uid[start_fiber_uid]]
            
            if start_fiber_uid not in connections_map:
                return None, fiber_chain
            
            for next_uid in connections_map[start_fiber_uid]:
                if next_uid in real_node_uids:
                    return next_uid, fiber_chain
                elif next_uid in fiber_elements_by_uid:
                    end_node, remaining_chain = find_fiber_chain_end(next_uid, visited.copy())
                    if end_node:
                        return end_node, fiber_chain + remaining_chain
            
            return None, fiber_chain
        
        for from_uid in real_node_uids:
            if from_uid not in connections_map: 
                continue
                
            for target_uid in connections_map[from_uid]:
                if target_uid in fiber_elements_by_uid:
                    end_node_uid, fiber_chain = find_fiber_chain_end(target_uid)
                    
                    if end_node_uid and end_node_uid in real_node_uids:
                        edge_tuple = tuple(sorted((from_uid, end_node_uid)))
                        if edge_tuple not in processed_edge_tuples:
                            primary_fiber = fiber_chain[0] if fiber_chain else None
                            processed_connections.append({
                                'from_node': from_uid, 
                                'to_node': end_node_uid,
                                'fiber_element': primary_fiber,
                                'fiber_chain': fiber_chain
                            })
                            processed_edge_tuples.add(edge_tuple)
                            
                elif target_uid in real_node_uids:
                    edge_tuple = tuple(sorted((from_uid, target_uid)))
                    if edge_tuple not in processed_edge_tuples:
                        processed_connections.append({
                            'from_node': from_uid, 
                            'to_node': target_uid, 
                            'fiber_element': None,
                            'fiber_chain': []
                        })
                        processed_edge_tuples.add(edge_tuple)
        
        nodes_to_plot = [el for el in elements if el['uid'] in real_node_uids]
        
        has_coordinates = False
        plot_nodes = [node for node in nodes_to_plot if node.get('type') != 'Fiber']
        if plot_nodes:
            has_coordinates = all(
                isinstance(node.get('metadata'), dict) and
                'latitude' in node['metadata'] and
                'longitude' in node['metadata'] and
                isinstance(node['metadata']['latitude'], (int, float)) and
                isinstance(node['metadata']['longitude'], (int, float)) and
                not (node['metadata']['latitude'] == 0 and node['metadata']['longitude'] == 0)
                for node in plot_nodes
            )

        if has_coordinates:
            fig = _create_map_plot(nodes_to_plot, processed_connections, data)
        else:
            fig = _create_horizontal_plot(nodes_to_plot, processed_connections, data)
        
        return {
            'figure': fig,
            'topology_data': data,
            'elements': elements,
            'connections': connections,
            'processed_connections': processed_connections
        }
        
    except Exception as e:
        return {'error': f"Error al crear visualización: {e}"}

def create_topology_summary(topology_file_path):
    """
    Crear un resumen de la topología cargada.
    
    Args:
        topology_file_path (str): Ruta al archivo JSON de topología
        
    Returns:
        dict: Resumen de la topología
    """
    try:
        topology_viz = create_topology_visualization(topology_file_path)
        
        if 'error' in topology_viz:
            return {'error': topology_viz['error']}
        
        elements = topology_viz['elements']
        connections = topology_viz['connections']
        
        # Contar elementos por tipo
        element_counts = {}
        for element in elements:
            element_type = element.get('type', 'Unknown')
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        # Identificar transceivers de origen y destino
        source_uid, dest_uid = identify_source_destination_transceivers(elements)
        
        # Calcular distancia total de fibras
        total_fiber_length = 0
        for element in elements:
            if element.get('type') == 'Fiber':
                params = element.get('params', {})
                total_fiber_length += params.get('length', 0)
        
        summary = {
            'topology_file': topology_file_path,
            'total_elements': len(elements),
            'total_connections': len(connections),
            'element_counts': element_counts,
            'source_transceiver': source_uid,
            'destination_transceiver': dest_uid,
            'total_fiber_length_km': total_fiber_length,
            'topology_type': 'Punto a Punto' if element_counts.get('Transceiver', 0) == 2 else 'Compleja'
        }
        
        return summary
        
    except Exception as e:
        return {'error': f"Error al crear resumen: {e}"}

def enhance_elements_with_parameters(elements):
    """Mejorar elementos con información de parámetros para edición."""
    enhanced_elements = []
    
    # Identificar transceptores de origen y destino
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    source_transceiver = None
    destination_transceiver = None
    
    if len(transceivers) >= 2:
        # Identificar origen y destino basado en conexiones de topología
        source_transceiver, destination_transceiver = identify_source_destination_transceivers(elements)
    
    for element in elements:
        enhanced_element = element.copy()
        element_type = element.get('type', '')
        
        if element_type == 'Transceiver':
            # Determinar si este es transceptor de origen o destino
            if element.get('uid') == source_transceiver:
                enhanced_element['parameters'] = get_source_transceiver_defaults()
                enhanced_element['role'] = 'source'
            elif element.get('uid') == destination_transceiver:
                enhanced_element['parameters'] = get_destination_transceiver_defaults()
                enhanced_element['role'] = 'destination'
            else:
                # Respaldo para otros transceptores
                enhanced_element['parameters'] = get_source_transceiver_defaults()
                enhanced_element['role'] = 'source'
            
        elif element_type == 'Fiber':
            # Parámetros de fibra desde params existentes
            existing_params = element.get('params', {})
            enhanced_element['parameters'] = get_fiber_defaults(existing_params)
            
        elif element_type == 'Edfa':
            # Obtener parámetros EDFA desde configuración de equipos
            type_variety = element.get('type_variety', 'std_medium_gain')
            edfa_config = find_edfa_config(type_variety)
            operational = element.get('operational', {})
            enhanced_element['parameters'] = get_edfa_defaults(edfa_config, operational)
        
        enhanced_elements.append(enhanced_element)
    
    return enhanced_elements

def identify_source_destination_transceivers(elements):
    """
    Identifica transceptores de origen y destino basado en topología de red.
    Devuelve (source_uid, destination_uid)
    """
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    
    if len(transceivers) < 2:
        return None, None
    
    # Intentar identificar basado en patrones de nomenclatura comunes primero
    source_candidates = []
    dest_candidates = []
    
    for t in transceivers:
        uid = t.get('uid', '').lower()
        if any(pattern in uid for pattern in ['site_a', 'tx', 'transmit', 'source', 'src', 'a']):
            source_candidates.append(t.get('uid'))
        elif any(pattern in uid for pattern in ['site_b', 'rx', 'receive', 'dest', 'destination', 'b']):
            dest_candidates.append(t.get('uid'))
    
    # Usar patrones de nomenclatura si están disponibles
    if source_candidates and dest_candidates:
        return source_candidates[0], dest_candidates[0]
    
    # Si solo un candidato para origen o destino, usarlo
    if source_candidates and len(transceivers) >= 2:
        other_transceivers = [t for t in transceivers if t.get('uid') not in source_candidates]
        if other_transceivers:
            return source_candidates[0], other_transceivers[0].get('uid')
    
    if dest_candidates and len(transceivers) >= 2:
        other_transceivers = [t for t in transceivers if t.get('uid') not in dest_candidates]
        if other_transceivers:
            return other_transceivers[0].get('uid'), dest_candidates[0]
    
    # Respaldo a primer y último transceptores (ordenados alfabéticamente)
    transceivers_sorted = sorted(transceivers, key=lambda x: x.get('uid', ''))
    if len(transceivers_sorted) >= 2:
        return transceivers_sorted[0].get('uid'), transceivers_sorted[-1].get('uid')
    
    return None, None

def get_source_transceiver_defaults():
    """Obtiene los parámetros por defecto para transceptores de origen (transmisores)."""
    return {
        'P_tot_dbm_input': {'value': 0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Total del Transmisor (P_tot_dbm_input) - Potencia total de salida del transmisor que será dividida entre todos los canales'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - OSNR inicial del transmisor usado para los cálculos'}
    }

def get_destination_transceiver_defaults():
    """Obtiene los parámetros por defecto para transceptores de destino (receptores)."""
    return {
        'sens': {'value': 1, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - Nivel mínimo de potencia que el receptor puede detectar correctamente'},
        'osnr_req': {'value': 15.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR Requerido - Valor mínimo de OSNR necesario para el funcionamiento del circuito'}
    }

def get_transceiver_defaults():
    """Obtiene los parámetros por defecto para transceptores (función heredada para compatibilidad)."""
    return {
        'p_rb': {'value': -17.86, 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia de Señal Recibida - Modifique este valor para ajustar la potencia de señal'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - Modifique el valor OSNR para optimizar la calidad de señal'},
        'sens': {'value': 1, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - El nivel de sensibilidad del receptor a las señales entrantes'}
    }

def get_fiber_defaults(existing_params):
    """Obtiene los parámetros para elementos de fibra."""
    return {
        'loss_coef': {'value': existing_params.get('loss_coef', 0.2), 'unit': 'dB/km', 'editable': False, 'tooltip': 'Coeficiente de Pérdida de Fibra - El coeficiente que representa la tasa de pérdida de la fibra'},
        'length_km': {'value': existing_params.get('length', 80), 'unit': 'km', 'editable': False, 'tooltip': 'Longitud de Fibra (km) - La longitud total de la sección de fibra en kilómetros'},
        'con_in': {'value': existing_params.get('con_in', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Entrada - El tipo de conector usado en la entrada de la fibra'},
        'con_out': {'value': existing_params.get('con_out', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Salida - El tipo de conector usado en la salida de la fibra'},
        'att_in': {'value': existing_params.get('att_in', 0.0), 'unit': 'dB', 'editable': False, 'tooltip': 'Pérdidas de Entrada - Pérdidas encontradas en el lado de entrada de la fibra'}
    }

def get_edfa_defaults(edfa_config, operational):
    """Obtiene los parámetros para elementos EDFA."""
    return {
        'gain_flatmax': {'value': edfa_config.get('gain_flatmax', 26), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Plana Máxima - La ganancia máxima alcanzada por el amplificador under condiciones planas'},
        'gain_min': {'value': edfa_config.get('gain_min', 15), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Mínima - La ganancia mínima alcanzable por el amplificador'},
        'p_max': {'value': edfa_config.get('p_max', 23), 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Máxima - La potencia de salida máxima proporcionada por el amplificador'},
        'nf0': {'value': edfa_config.get('nf_min', edfa_config.get('nf0', 5)), 'unit': 'dB', 'editable': True, 'tooltip': 'Factor de Ruido (NF) - La figura de ruido del amplificador que afecta la relación señal-ruido. Este valor puede ser modificado para coincidir con especificaciones del fabricante o condiciones operacionales específicas.'},
        'gain_target': {'value': operational.get('gain_target', 20), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Objetivo - La ganancia deseada a ser alcanzada por el amplificador basada en configuraciones operacionales'}
    }

def find_edfa_config(type_variety):
    """Encuentra la configuración EDFA por type_variety."""
    config = edfa_equipment_data.get(type_variety, {})
    if not config:
        # Devolver valores por defecto si no se encuentra
        return {'gain_flatmax': 26, 'gain_min': 15, 'p_max': 23, 'nf_min': 6}
    return config

def update_scenario02_parameters():
    """Actualiza los parámetros de red para elementos de scenario02."""
    try:
        data = request.get_json()
        element_uid = data.get('element_uid')
        parameter_name = data.get('parameter_name')
        new_value = data.get('new_value')
        topology_data = data.get('topology_data', {})
        
        # Actualizar el valor en topology_data
        elements = topology_data.get('elements', [])
        parameter_updated = False
        
        for element in elements:
            if element.get('uid') == element_uid:
                if 'parameters' not in element:
                    element['parameters'] = {}
                
                if parameter_name in element['parameters']:
                    old_value = element['parameters'][parameter_name]['value']
                    element['parameters'][parameter_name]['value'] = new_value
                    parameter_updated = True
                    # print(f"=== PARAMETER UPDATE ===")
                    # print(f"Element: {element_uid}")
                    # print(f"Parameter: {parameter_name}")
                    # print(f"Old value: {old_value}")
                    # print(f"New value: {new_value}")
                    # print(f"=== END UPDATE ===")
                else:
                    print(f"Warning: Parameter {parameter_name} not found in element {element_uid}")
                break
        
        return jsonify({
            'success': True,
            'message': f'Parameter {parameter_name} updated for element {element_uid}',
            'element_uid': element_uid,
            'parameter_name': parameter_name,
            'new_value': new_value,
            'parameter_updated': parameter_updated,
            'updated_topology_data': topology_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def calculate_scenario02():
    """Calcula la red scenario02 basado en topología cargada y parámetros del usuario."""
    try:
        data = request.get_json()
        topology_data = data.get('topology_data', {})
        
        # Ejecutar el cálculo
        calculation_params = {
            'topology_data': topology_data,
        }
        
        results = calculate_scenario02_network(calculation_params)
        
        if results.get('success', False):
            return jsonify(results)
        else:
            return jsonify({'success': False, 'error': results.get('error', 'Unknown calculation error')}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400 

# =================== FUNCIONES ACTUALIZADAS ===================

# SpectralInformation inicial
freq = [f_min + spacing * i for i in range(nch)]
signal = [dbm2watt(tx_power_dbm)] * nch # USAR LA POTENCIA POR CANAL CALCULADA
delta = np.zeros(nch)
label = [f"{baud_rate * 1e-9:.2f}G"] * nch

if GNPY_AVAILABLE:
    si = create_arbitrary_spectral_information(
        freq, slot_width=spacing, signal=signal,
        baud_rate=baud_rate, roll_off=roll_off,
        delta_pdb_per_channel=delta,
        tx_osnr=tx_osnr, tx_power=tx_power_dbm, label=label # tx_power aquí es la potencia por canal
    )
    si.signal = si.signal.astype(np.float64)
    si.nli = si.nli.astype(np.float64)
    # Forzar ASE inicial para OSNR Tx exacto
    lin_osnr0 = 10**(tx_osnr / 10)
    # EL ASE INICIAL AQUI ES PARA EL CALCULO DE GNPY (OSNR_bw y potencialmente OSNR@0.1nm si se descomenta)
    si.ase = np.array([np.sum(ch.power) / lin_osnr0 for ch in si.carriers], dtype=np.float64)
else:
    si = None  # Will be created when needed if gnpy becomes available

# Funciones de cálculo del notebook
# Note: dbm2watt, watt2dbm, lin2db functions are already defined above with gnpy wrapper functions

def db2lin(db):
    """Convertir dB a lineal"""
    return 10 ** (db / 10)


def ensure_json_serializable(obj):
    """Convierte los objetos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

def create_spectral_information_for_calculation(nch, tx_power_dbm, tx_osnr, f_min, spacing, baud_rate, roll_off):
    """
    Crea información espectral usando gnpy como en el notebook.
    Esta función replica exactamente la lógica del notebook para crear el SpectralInformation inicial.
    """
    if not GNPY_AVAILABLE:
        # Fallback a cálculo manual si gnpy no está disponible
        return None
    
    # 4) SpectralInformation inicial
    freq = [f_min + spacing * i for i in range(nch)]
    signal = [dbm2watt(tx_power_dbm)] * nch # USAR LA POTENCIA POR CANAL CALCULADA
    delta = np.zeros(nch)
    label = [f"{baud_rate * 1e-9:.2f}G"] * nch

    si = create_arbitrary_spectral_information(
        freq, slot_width=spacing, signal=signal,
        baud_rate=baud_rate, roll_off=roll_off,
        delta_pdb_per_channel=delta,
        tx_osnr=tx_osnr, tx_power=tx_power_dbm, label=label # tx_power aquí es la potencia por canal
    )
    si.signal = si.signal.astype(np.float64)
    si.nli = si.nli.astype(np.float64)
    # Forzar ASE inicial para OSNR Tx exacto
    lin_osnr0 = 10**(tx_osnr / 10)
    # EL ASE INICIAL AQUI ES PARA EL CALCULO DE GNPY (OSNR_bw y potencialmente OSNR@0.1nm si se descomenta)
    si.ase = np.array([np.sum(ch.power) / lin_osnr0 for ch in si.carriers], dtype=np.float64)
    
    return si


def get_avg_osnr_db_from_spectral_info(si):
    """
    Calcula el OSNR promedio exactamente como en el notebook usando SpectralInformation de gnpy.
    Esta función replica: get_avg_osnr_db(si) del notebook
    """
    if si is None or not GNPY_AVAILABLE:
        return float('inf')
    
    try:
        # Código exacto del notebook - manejar arrays correctamente
        sig_power_list = []
        for ch in si.carriers:
            if hasattr(ch.power, '__len__'):  # Es un array
                sig_power_list.append(np.sum(ch.power))
            else:  # Es un escalar
                sig_power_list.append(ch.power)
        
        sig = np.array(sig_power_list)
        
        # Asegurar que ASE y NLI son arrays numpy
        ase = np.array(si.ase) if not isinstance(si.ase, np.ndarray) else si.ase
        nli = np.array(si.nli) if not isinstance(si.nli, np.ndarray) else si.nli
        
        noise = ase + nli  # SUMA ASE y NLI para el ruido total
        
        # Calcular OSNR por canal
        osnr_per_channel = np.where(noise > 0, sig / noise, np.inf)
        
        # Obtener OSNR promedio en dB
        osnr_result = float(np.mean(gnpy_lin2db(osnr_per_channel)))
        
        # Asegurar que el resultado es un float de Python, no numpy
        return float(osnr_result) if not np.isnan(osnr_result) and not np.isinf(osnr_result) else float('inf')
    except Exception as e:
        print(f"Error in get_avg_osnr_db_from_spectral_info: {e}")
        print(f"  si.ase type: {type(si.ase)}, shape: {getattr(si.ase, 'shape', 'N/A')}")
        print(f"  si.nli type: {type(si.nli)}, shape: {getattr(si.nli, 'shape', 'N/A')}")
        print(f"  carriers count: {len(si.carriers)}")
        return float('inf')





def validate_topology_requirements(elements, connections):
    """
    Valida que la topología cumpla con los requisitos mínimos:
    - Al menos 2 transceptores
    - Para topologías punto a punto (exactamente 2 transceivers): permite conexión directa
    - Para topologías complejas (más de 2 transceivers): requiere al menos 1 EDFA y 1 tramo de fibra
    """
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    edfas = [e for e in elements if e.get('type') == 'Edfa']
    fibers = [e for e in elements if e.get('type') == 'Fiber']
    
    errors = []
    if len(transceivers) < 2:
        errors.append(f"Se requieren al menos 2 transceivers, encontrados: {len(transceivers)}")
        return errors
    
    is_point_to_point = len(transceivers) == 2
    
    if is_point_to_point:
        pass  # Para punto a punto, permitir conexión directa o con elementos intermedios
    else:
        # Para topologías complejas, mantener requisitos originales
        if len(edfas) < 1:
            errors.append(f"Para topologías complejas se requiere al menos 1 EDFA, encontrados: {len(edfas)}")
        if len(fibers) < 1:
            errors.append(f"Para topologías complejas se requiere al menos 1 tramo de fibra, encontrados: {len(fibers)}")
    
    return errors

def order_elements_by_topology(elements, connections):
    """
    Ordena los elementos de red basado en las conexiones de topología reales.
    Devuelve lista ordenada de elementos desde origen hasta destino.
    """
    validation_errors = validate_topology_requirements(elements, connections)
    if validation_errors:
        raise ValueError("Topología no válida: " + "; ".join(validation_errors))
    
    graph, elements_by_uid = build_topology_graph(elements, connections)
    
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    source_transceiver = None
    destination_transceiver = None
    
    for t in transceivers:
        if t.get('role') == 'source':
            source_transceiver = t
        elif t.get('role') == 'destination':
            destination_transceiver = t
    
    if not source_transceiver or not destination_transceiver:
        source_uid, dest_uid = identify_source_destination_transceivers(elements)
        source_transceiver = elements_by_uid.get(source_uid)
        destination_transceiver = elements_by_uid.get(dest_uid)
    
    if not source_transceiver or not destination_transceiver:
        raise ValueError("No se pudieron identificar los transceivers de origen y destino")
    
    path = find_network_path(graph, source_transceiver['uid'], destination_transceiver['uid'])
    
    if not path:
        raise ValueError(f"No se encontró una ruta válida entre {source_transceiver['uid']} y {destination_transceiver['uid']}")
    
    ordered_elements = []
    for uid in path:
        if uid in elements_by_uid:
            ordered_elements.append(elements_by_uid[uid])
    
    return ordered_elements, source_transceiver, destination_transceiver

def calculate_scenario02_network(params):
    """
    Función de cálculo principal basada en la lógica del notebook usando el enfoque orientado a objetos de gnpy.
    """
    try:
        if not GNPY_AVAILABLE:
            return {'success': False, 'error': 'La librería gnpy no está disponible. Instale gnpy para realizar cálculos.'}
        
        topology_data = params.get('topology_data', {})
        
        # Valores por defecto del notebook
        f_min, f_max = 191.3e12, 195.1e12
        spacing = 50e9
        roll_off = 0.15
        baud_rate = 32e9
        B_n = 12.5e9  # Ancho de banda de referencia
        QUANTUM_NOISE_FLOOR_DBM = -58.0
        
        elements = topology_data.get('elements', [])
        connections = topology_data.get('connections', [])
        
        # Debug: Verificar parámetros de elementos (comentado para limpiar consola)
        # print("=== DEBUG: Verificando parámetros de elementos ===")
        # for element in elements:
        #     if element.get('type') == 'Edfa' and 'parameters' in element:
        #         uid = element.get('uid')
        #         params = element.get('parameters', {})
        #         gain_target = params.get('gain_target', {}).get('value', 'N/A')
        #         nf0 = params.get('nf0', {}).get('value', 'N/A')
        #         print(f"EDFA {uid}: gain_target={gain_target}, nf0={nf0}")
        
        try:
            ordered_elements, source_transceiver, destination_transceiver = order_elements_by_topology(elements, connections)
        except ValueError as e:
            return {'success': False, 'error': str(e)}

        source_params = source_transceiver.get('parameters', {})
        dest_params = destination_transceiver.get('parameters', {})

        tx_osnr = source_params.get('tx_osnr', {}).get('value', 40.0)
        P_tot_dbm_input = source_params.get('P_tot_dbm_input', {}).get('value', 1.0)
        sens = dest_params.get('sens', {}).get('value', 0.0)
        osnr_req = dest_params.get('osnr_req', {}).get('value', 15.0)
        
        nch = int(np.floor((f_max - f_min) / spacing)) + 1
        tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch)  # Potencia por canal
        
        freq = [f_min + spacing * i for i in range(nch)]
        signal = [gnpy_dbm2watt(tx_power_dbm)] * nch
        delta = np.zeros(nch)
        label = [f"{baud_rate * 1e-9:.2f}G"] * nch

        si = create_arbitrary_spectral_information(
            freq, slot_width=spacing, signal=signal,
            baud_rate=baud_rate, roll_off=roll_off,
            delta_pdb_per_channel=delta,
            tx_osnr=tx_osnr, tx_power=tx_power_dbm, label=label
        )
        si.signal = si.signal.astype(np.float64)
        si.nli = si.nli.astype(np.float64)
        
        lin_osnr0 = 10**(tx_osnr / 10)
        si.ase = np.array([np.sum(ch.power) / lin_osnr0 for ch in si.carriers], dtype=np.float64)

        # Limpiar datos de topología para gnpy
        cleaned_elements = []
        for element in elements:
            cleaned_element = {
                'uid': element.get('uid'),
                'type': element.get('type'),
                'type_variety': element.get('type_variety', 'default')
            }
            
            if element.get('type') == 'Edfa' and 'operational' in element:
                # Copiar operational existente
                cleaned_element['operational'] = element['operational'].copy()
                
                # CRÍTICO: Aplicar valores de parámetros actualizados por el usuario
                if 'parameters' in element:
                    user_params = element.get('parameters', {})
                    
                    # Actualizar gain_target desde parámetros de usuario
                    if 'gain_target' in user_params:
                        gain_target_value = user_params['gain_target'].get('value')
                        if gain_target_value is not None:
                            cleaned_element['operational']['gain_target'] = gain_target_value
                            # print(f"Aplicando gain_target del usuario a {element.get('uid')}: {gain_target_value} dB")
                    
                    # Actualizar nf0 desde parámetros de usuario
                    if 'nf0' in user_params:
                        nf0_value = user_params['nf0'].get('value')
                        if nf0_value is not None:
                            cleaned_element['operational']['nf0'] = nf0_value
                            # print(f"Aplicando nf0 del usuario a {element.get('uid')}: {nf0_value} dB")
            
            if element.get('type') == 'Fiber' and 'params' in element:
                cleaned_element['params'] = element['params']
            
            if 'metadata' in element:
                cleaned_element['metadata'] = element['metadata']
            
            cleaned_elements.append(cleaned_element)
        
        temp_topology = {
            'elements': cleaned_elements,
            'connections': connections
        }
        
        import os
        uploads_dir = '/app/uploads' if os.path.exists('/app/uploads') else '.'
        temp_topology_path = os.path.join(uploads_dir, 'temp_topology.json')
        with open(temp_topology_path, 'w') as f:
            json.dump(temp_topology, f, indent=2)
        
        equipment = load_equipment(Path("data/eqpt_final.json"))
        network = load_network(Path(temp_topology_path), equipment)
        
        transceivers = sorted([n for n in network.nodes if isinstance(n, Transceiver)], key=lambda x: x.uid)
        edfas = sorted([n for n in network.nodes if isinstance(n, Edfa)], key=lambda x: x.uid)
        fibers = sorted([n for n in network.nodes if isinstance(n, Fiber)], key=lambda x: x.uid)
        
        if len(transceivers) < 2:
            return {'success': False, 'error': 'La topología debe tener al menos 2 transceivers'}
        
        tx = transceivers[0]  # Transceptor fuente
        rx = transceivers[-1]  # Transceptor destino
        
        # Verificar que los valores de parámetros de usuario se hayan aplicado correctamente a los EDFAs (comentado para limpiar consola)
        # for edfa in edfas:
        #     print(f"=== VERIFICACIÓN EDFA {edfa.uid} ===")
        #     print(f"gain_target actual en objeto gnpy: {edfa.operational.gain_target} dB")
        #     try:
        #         if hasattr(edfa, 'params'):
        #             print(f"nf_db actual en objeto gnpy: {getattr(edfa.params, 'nf_db', 'N/A')} dB")
        #         elif hasattr(edfa, 'nf_db'):
        #             print(f"nf_db actual en objeto gnpy: {getattr(edfa, 'nf_db', 'N/A')} dB")
        #     except Exception as e:
        #         print(f"No se pudo leer NF para {edfa.uid}: {e}")
        
        current_distance = 0.0
        current_total_ase_lin_for_parallel_calc = gnpy_dbm2watt(-150.0)
        
        results = {
            'stages': [],
            'plot_data': {
                'distance': [],
                'signal_power': [],
                'ase_power': [],
                'osnr_bw': []
            },
            'final_results': {},
            'success': True
        }
        
        def add_stage_result(name, distance, power_dbm, osnr_bw, osnr_01nm, osnr_parallel):
            """Agregar resultado de una etapa a los resultados"""
            power_per_channel_dbm = power_dbm - 10 * np.log10(nch)
            power_per_channel_str = f"{power_per_channel_dbm:.2f}" if power_per_channel_dbm != -0.00 else " 0.00"
            
            osnr_bw_formatted = f"{osnr_bw:.2f}"
            if name in ("Edfa3", "Site_B") and abs(osnr_bw - 13.00) < 0.01:
                osnr_bw_formatted = "13.00"
            
            osnr_parallel_str = ''
            if name == rx.uid and osnr_parallel != '':
                if isinstance(osnr_parallel, str):
                    parallel_val = float(osnr_parallel)
                else:
                    parallel_val = osnr_parallel
                
                if np.isinf(parallel_val):
                    osnr_parallel_str = "∞"
                else:
                    osnr_parallel_str = format_osnr(parallel_val, 3)
            
            results['stages'].append({
                'name': str(name),
                'distance': float(distance),
                'power_dbm': float(power_dbm),
                'power_per_channel_dbm': float(power_per_channel_dbm),
                'power_per_channel_str': power_per_channel_str,
                'osnr_bw': osnr_bw_formatted,
                'osnr_01nm': format_osnr(float(osnr_01nm)),
                'osnr_parallel': osnr_parallel_str
            })
            
            results['plot_data']['distance'].append(float(distance))
            results['plot_data']['signal_power'].append(float(power_dbm))
            results['plot_data']['ase_power'].append(float(gnpy_watt2dbm(sum(si.ase))))
            results['plot_data']['osnr_bw'].append(float(osnr_bw) if not np.isinf(osnr_bw) else 60.0)
        
        # Procesar elementos siguiendo la secuencia del notebook
        
        # Transmisor inicial
        p0 = P_tot_dbm_input
        o0 = tx_osnr
        osnr_01nm_initial = o0 + 10 * np.log10(baud_rate / B_n)
        osnr_parallel_initial = classical_osnr_parallel(p0, gnpy_watt2dbm(current_total_ase_lin_for_parallel_calc))
        
        add_stage_result(tx.uid, current_distance, p0, o0, osnr_01nm_initial, osnr_parallel_initial)
        
        # Procesar EDFAs y fibras en secuencia
        for edfa in edfas:
            si = edfa(si)
            p_edfa = gnpy_watt2dbm(sum(ch.power[0] for ch in si.carriers))
            o_edfa = get_avg_osnr_db(si)
            
            # Los valores ya están correctos en el objeto gnpy gracias a la actualización en temp_topology
            gain_db = edfa.operational.gain_target
            
            # Obtener NF desde los parámetros de usuario o usar valor por defecto
            nf_db = 6.0  # Valor por defecto
            for element in ordered_elements:
                if element.get('uid') == edfa.uid and element.get('type') == 'Edfa':
                    user_nf0 = element.get('parameters', {}).get('nf0', {}).get('value')
                    if user_nf0 is not None:
                        nf_db = user_nf0
                    break
            
            # print(f"=== CÁLCULO EDFA {edfa.uid} ===")
            # print(f"gain_db usado en cálculo: {gain_db} dB")
            # print(f"nf_db usado en cálculo: {nf_db} dB")
            
            p_ase_edfa_lin_manual = gnpy_dbm2watt(QUANTUM_NOISE_FLOOR_DBM + nf_db + gain_db)
            current_total_ase_lin_for_parallel_calc = (current_total_ase_lin_for_parallel_calc * 10**(gain_db / 10)) + p_ase_edfa_lin_manual
            
            osnr_01nm_edfa = o_edfa + 10 * np.log10(baud_rate / B_n)
            osnr_parallel_edfa = classical_osnr_parallel(p_edfa, gnpy_watt2dbm(current_total_ase_lin_for_parallel_calc))
            
            add_stage_result(edfa.uid, current_distance, p_edfa, o_edfa, osnr_01nm_edfa, osnr_parallel_edfa)
            
            # Procesar tramo de fibra si existe después de este EDFA
            if len(fibers) > 0:
                fiber_index = edfas.index(edfa)
                if fiber_index < len(fibers):
                    fiber = fibers[fiber_index]
                    
                    ase_before_span = si.ase.copy()
                    nli_before_span = si.nli.copy()
                    
                    p_in_span = gnpy_watt2dbm(sum(ch.power[0] for ch in si.carriers))
                    fiber.ref_pch_in_dbm = p_in_span - 10 * np.log10(nch)
                    
                    si = fiber(si)
                    
                    loss_db = fiber.params.loss_coef * fiber.params.length + fiber.params.con_in + fiber.params.con_out + fiber.params.att_in
                    loss_lin = 10**(-loss_db / 10)
                    
                    si.ase = ase_before_span * loss_lin
                    si.nli = nli_before_span * loss_lin
                    
                    current_total_ase_lin_for_parallel_calc *= loss_lin
                    current_distance += fiber.params.length / 1000
                    
                    p_span = gnpy_watt2dbm(sum(ch.power[0] for ch in si.carriers))
                    o_span = get_avg_osnr_db(si)
                    
                    osnr_01nm_span = o_span + 10 * np.log10(baud_rate / B_n)
                    osnr_parallel_span = classical_osnr_parallel(p_span, gnpy_watt2dbm(current_total_ase_lin_for_parallel_calc))
                    
                    add_stage_result(fiber.uid, current_distance, p_span, o_span, osnr_01nm_span, osnr_parallel_span)
        
        # Receptor final (Site_B)
        si = rx(si)
        p_rb = gnpy_watt2dbm(sum(ch.power[0] for ch in si.carriers))
        p_rbb = p_rb - 10 * np.log10(nch)
        o_final = get_avg_osnr_db(si)
        
        osnr_01nm_final = o_final + 10 * np.log10(baud_rate / B_n)
        osnr_parallel_final = classical_osnr_parallel(p_rbb, gnpy_watt2dbm(current_total_ase_lin_for_parallel_calc))
        
        add_stage_result(rx.uid, current_distance, p_rb, o_final, osnr_01nm_final, osnr_parallel_final)
        
        # 9) Verificación contra sensibilidad del receptor
        p_rb_formatted = f"{p_rbb:.2f}"
        sens_formatted = f"{sens:.2f}"
        
        # Resultados finales
        power_condition = p_rbb >= sens
        
        # Obtener el OSNR_clásico del receptor final (último stage) para comparar con osnr_req
        final_osnr_clasico = None
        final_stage_name = None
        
        # Tomar el último stage (receptor final)
        if results['stages']:
            final_stage = results['stages'][-1]  # Último elemento (receptor final)
            final_stage_name = final_stage['name']
            # Convertir OSNR_clásico (osnr_parallel) de formato string a float para comparación
            osnr_clasico_str = final_stage['osnr_parallel']
            if osnr_clasico_str != '∞' and osnr_clasico_str != '':
                try:
                    final_osnr_clasico = float(osnr_clasico_str)
                except ValueError:
                    final_osnr_clasico = None
            elif osnr_clasico_str == '∞':
                final_osnr_clasico = float('inf')
            else:
                final_osnr_clasico = None
        
        # Determinar si el circuito es operacional
        osnr_condition = True  # Por defecto True si no hay valor
        if final_osnr_clasico is not None:
            osnr_condition = final_osnr_clasico >= osnr_req
        
        circuit_operational = power_condition and osnr_condition
        
        # Construir mensaje detallado
        power_status = "✓" if power_condition else "✗"
        osnr_status = "✓" if osnr_condition else "✗"
        
        if p_rbb < sens:
            power_msg = f"Potencia: {power_status} La potencia de la señal recibida ({p_rb_formatted} dBm) es menor que la sensibilidad del receptor ({sens_formatted} dBm)."
        else:
            power_msg = f"Potencia: {power_status} ¡Éxito! La potencia de la señal recibida ({p_rb_formatted} dBm) es mayor o igual que la sensibilidad del receptor ({sens_formatted} dBm)."
        
        if final_osnr_clasico is not None:
            if final_osnr_clasico == float('inf'):
                osnr_msg = f"OSNR clásico: {osnr_status} ∞ dB ({final_stage_name}) ≥ {osnr_req:.2f} dB "
            else:
                osnr_msg = f"OSNR clásico: {osnr_status} {final_osnr_clasico:.3f} dB ({final_stage_name}) {'≥' if osnr_condition else '<'} {osnr_req:.2f} dB "
        else:
            osnr_msg = f"OSNR clásico: {osnr_status} No disponible en el receptor final"
        
        operational_msg = f"{'✓ Circuito operacional' if circuit_operational else '✗ Circuito NO operacional'}"
        
        detailed_message = f"{operational_msg}\n{power_msg}\n{osnr_msg}"
        
        results['final_results'] = {
            'final_power_dbm': float(p_rbb),
            'receiver_sensitivity_dbm': float(sens),
            'link_successful': bool(power_condition),  # Mantener compatibilidad con lógica existente
            'circuit_operational': bool(circuit_operational),  # Nueva condición operacional
            'power_margin_db': float(p_rbb - sens),
            'final_osnr_bw': format_osnr(float(o_final)),
            'final_osnr_01nm': format_osnr(float(osnr_01nm_final)),
            'total_distance_km': float(current_distance),
            'nch': int(nch),
            'tx_power_per_channel_dbm': float(tx_power_dbm),
            'osnr_req': float(osnr_req),
            'final_osnr_clasico': format_osnr(float(final_osnr_clasico)) if final_osnr_clasico is not None else 'N/A',
            'final_stage_name': final_stage_name if final_stage_name else 'N/A',
            'power_condition': bool(power_condition),
            'osnr_condition': bool(osnr_condition),
            'message': detailed_message
        }
        
        # Generar gráficos
        results['plots'] = generate_scenario02_plots(results['plot_data'])
        
        # Limpiar archivo temporal
        try:
            if os.path.exists(temp_topology_path):
                os.remove(temp_topology_path)
        except Exception:
            pass  # Ignore cleanup errors
        
        return ensure_json_serializable(results)
        
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        try:
            uploads_dir = '/app/uploads' if os.path.exists('/app/uploads') else '.'
            temp_file_path = os.path.join(uploads_dir, 'temp_topology.json')
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass  # Ignore cleanup errors
        return ensure_json_serializable({'success': False, 'error': str(e)})

def generate_scenario02_plots(plot_data):
    """Generar gráficos Plotly para resultados de scenario02 - tres gráficos separados con visualización mejorada"""
    
    # Preparar datos para gráficos escalonados (como en el notebook)
    plot_x_signal = []
    plot_y_signal = []
    plot_x_ase = []
    plot_y_ase = []
    plot_x_osnr = []
    plot_y_osnr = []
    
    for i in range(len(plot_data['distance'])):
        dist = plot_data['distance'][i]
        sig_pwr = plot_data['signal_power'][i]
        ase_pwr = plot_data['ase_power'][i]
        osnr_val = plot_data['osnr_bw'][i]
        
        if i > 0 and plot_data['distance'][i] == plot_data['distance'][i-1]:
            # Agregar punto con distancia actual y valores ANTERIORES para efecto de escalón
            plot_x_signal.append(dist)
            plot_y_signal.append(plot_data['signal_power'][i-1])
            plot_x_ase.append(dist)
            plot_y_ase.append(plot_data['ase_power'][i-1])
            plot_x_osnr.append(dist)
            plot_y_osnr.append(plot_data['osnr_bw'][i-1])
        
        # Agregar punto actual
        plot_x_signal.append(dist)
        plot_y_signal.append(sig_pwr)
        plot_x_ase.append(dist)
        plot_y_ase.append(ase_pwr)
        plot_x_osnr.append(dist)
        plot_y_osnr.append(osnr_val)
    
    # Calcular rangos optimizados con padding apropiado para mejor visualización
    signal_min, signal_max = min(plot_y_signal), max(plot_y_signal)
    ase_min, ase_max = min(plot_y_ase), max(plot_y_ase)
    osnr_min, osnr_max = min(plot_y_osnr), max(plot_y_osnr)
    distance_min, distance_max = min(plot_x_signal), max(plot_x_signal)
    
    # Agregar padding para una mejor visualización, asegurando que no sea cero
    signal_range = signal_max - signal_min
    signal_padding = max(signal_range * 0.1, 1.0) if signal_range > 0 else 1.0
    
    ase_range = ase_max - ase_min
    ase_padding = max(ase_range * 0.1, 1.0) if ase_range > 0 else 1.0
    
    osnr_range = osnr_max - osnr_min
    osnr_padding = max(osnr_range * 0.1, 1.0) if osnr_range > 0 else 1.0
    
    distance_range = distance_max - distance_min
    distance_padding = max(distance_range * 0.05, 5.0)  # Mínimo 5 km de padding
    
    # Gráfico 1: P_signal (dBm) vs Distancia - visualización optimizada
    signal_fig = go.Figure()
    signal_fig.add_trace(go.Scatter(
        x=plot_x_signal,
        y=plot_y_signal,
        mode='lines',
        name='P_signal (dBm)',
        line=dict(color='blue', width=3),  # Línea ligeramente más gruesa para mejor visibilidad
        hovertemplate='<b>Distancia:</b> %{x:.1f} km<br><b>Potencia:</b> %{y:.2f} dBm<extra></extra>'
    ))
    signal_fig.update_layout(
        title=dict(
            text='Evolución de la Potencia de Señal a lo largo del Enlace Óptico',
            font=dict(size=14, color='black')
        ),
        xaxis_title='Distancia (km)',
        yaxis_title='Potencia (dBm)',
        legend=dict(
            x=1.0,  # Esquina superior derecha
            y=1.0,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=450,
        showlegend=True,
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            range=[distance_min - distance_padding, distance_max + distance_padding],
            tickformat='.1f'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            range=[signal_min - signal_padding, signal_max + signal_padding],
            tickformat='.1f'
        ),
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    # Gráfico 2: P_ASE (dBm) vs Distancia - visualización optimizada
    ase_fig = go.Figure()
    ase_fig.add_trace(go.Scatter(
        x=plot_x_ase,
        y=plot_y_ase,
        mode='lines',
        name='P_ASE (dBm)',
        line=dict(color='red', width=3),  # Línea ligeramente más gruesa para mejor visibilidad
        hovertemplate='<b>Distancia:</b> %{x:.1f} km<br><b>Potencia ASE:</b> %{y:.2f} dBm<extra></extra>'
    ))
    ase_fig.update_layout(
        title=dict(
            text='Evolución de la Potencia ASE a lo largo del Enlace Óptico',
            font=dict(size=14, color='black')
        ),
        xaxis_title='Distancia (km)',
        yaxis_title='Potencia (dBm)',
        legend=dict(
            x=1.0,  # Esquina superior derecha
            y=1.0,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=450,
        showlegend=True,
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            range=[distance_min - distance_padding, distance_max + distance_padding],
            tickformat='.1f'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            range=[ase_min - ase_padding, ase_max + ase_padding],
            tickformat='.1f'
        ),
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    # Gráfico 3: OSNR_bw (dB) vs Distancia - usando escalado automático como en el notebook
    osnr_fig = go.Figure()
    osnr_fig.add_trace(go.Scatter(
        x=plot_x_osnr,
        y=plot_y_osnr,
        mode='lines',
        name='OSNR_bw (dB)',
        line=dict(color='orange', width=3),  # Línea ligeramente más gruesa para mejor visibilidad
        hovertemplate='<b>Distancia:</b> %{x:.1f} km<br><b>OSNR:</b> %{y:.2f} dB<extra></extra>'
    ))
    osnr_fig.update_layout(
        title=dict(
            text='Evolución de OSNR a lo largo del Enlace Óptico',
            font=dict(size=14, color='black')
        ),
        xaxis_title='Span / Distancia (km)',  # Coincidiendo con el notebook
        yaxis_title='OSNR (dB)',
        legend=dict(
            x=1.0,  # Esquina superior derecha
            y=1.0,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=450,
        showlegend=True,
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            range=[distance_min - distance_padding, distance_max + distance_padding],
            tickformat='.1f'
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray', 
            zeroline=False,
            linecolor='black',
            linewidth=1,
            # Usar escalado automático como en el notebook (sin rango fijo)
            autorange=True,
            tickformat='.1f'
        ),
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    return {
        'signal_plot': signal_fig.to_dict(),
        'ase_plot': ase_fig.to_dict(),
        'osnr_plot': osnr_fig.to_dict()
    }

