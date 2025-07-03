import plotly.graph_objects as go
import json
import os
import numpy as np
from flask import jsonify, request
from pathlib import Path

# Check if gnpy is available
try:
    from gnpy.tools.json_io import load_equipment, load_network
    from gnpy.core.info import create_arbitrary_spectral_information
    from gnpy.core.utils import dbm2watt as gnpy_dbm2watt_orig, watt2dbm as gnpy_watt2dbm_orig, lin2db as gnpy_lin2db_orig
    from gnpy.core.elements import Transceiver, Fiber, Edfa
    GNPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: gnpy not available - {e}")
    GNPY_AVAILABLE = False
    gnpy_dbm2watt_orig = None
    gnpy_watt2dbm_orig = None
    gnpy_lin2db_orig = None

# Utility conversion functions that work with or without gnpy
def dbm2watt(dbm):
    """Convert dBm to Watts"""
    return 10 ** ((dbm - 30) / 10)

def watt2dbm(watt):
    """Convert Watts to dBm"""
    if watt <= 0:
        return -float('inf')
    return 30 + 10 * np.log10(watt)

def lin2db(lin):
    """Convert linear to dB"""
    if lin <= 0:
        return -float('inf')
    return 10 * np.log10(lin)

# Wrapper functions that use gnpy versions when available, fallback to local ones
def gnpy_dbm2watt(dbm):
    """Convert dBm to Watts using gnpy if available, otherwise local function"""
    if GNPY_AVAILABLE and gnpy_dbm2watt_orig:
        return gnpy_dbm2watt_orig(dbm)
    return dbm2watt(dbm)

def gnpy_watt2dbm(watt):
    """Convert Watts to dBm using gnpy if available, otherwise local function"""
    if GNPY_AVAILABLE and gnpy_watt2dbm_orig:
        return gnpy_watt2dbm_orig(watt)
    return watt2dbm(watt)

def gnpy_lin2db(lin):
    """Convert linear to dB using gnpy if available, otherwise local function"""
    if GNPY_AVAILABLE and gnpy_lin2db_orig:
        return gnpy_lin2db_orig(lin)
    return lin2db(lin)

# 1) PARÁMETROS
f_min, f_max = 191.3e12, 195.1e12
spacing = 50e9
roll_off = 0.15
tx_osnr = 45  # dB inicial, este valor es para los cálculos de gnpy de OSNR_bw
baud_rate = 32e9
B_n=12.5e9 #no sea modificable

# Calcular el número de canales antes de las entradas de usuario
nch = int(np.floor((f_max - f_min) / spacing)) + 1

# Cargar la configuración de equipos
EQPT_CONFIG_PATH = 'versionamientos/eqpt_config.json'
edfa_equipment_data = {}
if os.path.exists(EQPT_CONFIG_PATH):
    with open(EQPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        full_eqpt_config = json.load(f)
        if 'Edfa' in full_eqpt_config:
            for edfa_spec in full_eqpt_config['Edfa']:
                edfa_equipment_data[edfa_spec['type_variety']] = edfa_spec
else:
    print(f"Warning: Equipment configuration file not found at {EQPT_CONFIG_PATH}")


# 2) Default parameters (can be overridden by web interface)
sens = -25.0  # Default sensitivity in dBm
P_tot_dbm_input = 15.0  # Default total power in dBm

# Calcular la potencia por canal a partir de la potencia total ingresada
tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch) # Potencia POR CANAL en dBm


# 3) Helpers OSNR
def get_avg_osnr_db(si):
    sig = np.array([np.sum(ch.power) for ch in si.carriers])
    noise = si.ase + si.nli # SUMA ASE y NLI para el ruido total
    return float(np.mean(lin2db(np.where(noise > 0, sig / noise, np.inf))))

# Función get_avg_osnr_01nm_db comentada
# def get_avg_osnr_01nm_db(si):
#     return get_avg_osnr_db(si) + 10 * np.log10(si.baud_rate[0] / B_n)

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


# =================== TOPOLOGY VISUALIZATION FUNCTIONS ===================

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
        tooltip_html += "<hr><b>Operational Params:</b><br>"
        tooltip_html += f"&nbsp;&nbsp;gain_target: {op.get('gain_target', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;tilt_target: {op.get('tilt_target', 'N/A')} dB<br>"
        tooltip_html += f"&nbsp;&nbsp;out_voa: {op.get('out_voa', 'N/A')} dB<br>"
        tooltip_html += "<hr><b>Equipment Specs:</b><br>"
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
        # Fibra única, usar tooltip existente
        return get_element_tooltip_text(fiber_chain[0], edfa_specs)
    
    # Múltiples fibras en cadena
    tooltip_html = f"<b>Fiber Chain ({len(fiber_chain)} spans):</b><br>"
    total_length = 0
    
    for i, fiber in enumerate(fiber_chain, 1):
        uid = fiber.get('uid', 'N/A')
        params = fiber.get('params', {})
        length = params.get('length', 0)
        loss_coef = params.get('loss_coef', 0.2)
        total_length += length
        
        tooltip_html += f"<hr><b>Span {i}: {uid}</b><br>"
        tooltip_html += f"&nbsp;&nbsp;length: {length} km<br>"
        tooltip_html += f"&nbsp;&nbsp;loss_coef: {loss_coef} dB/km<br>"
        tooltip_html += f"&nbsp;&nbsp;total_loss: {(length * loss_coef):.2f} dB<br>"
    
    tooltip_html += f"<hr><b>Chain Summary:</b><br>"
    tooltip_html += f"&nbsp;&nbsp;Total Length: {total_length} km<br>"
    tooltip_html += f"&nbsp;&nbsp;Total Spans: {len(fiber_chain)}<br>"
        
    return tooltip_html


def get_node_styles_and_tooltips(nodes_to_plot, edfa_specs):
    """Obtiene estilos y tooltips para nodos en el gráfico."""
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


def identify_source_destination_transceivers(elements):
    """
    Identificar transceptores de origen y destino basado en topología de red.
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
        
        # Procesar elementos para visualización
        elements_by_uid = {el['uid']: el for el in elements}
        real_node_uids = {uid for uid, el in elements_by_uid.items() if el.get('type') != 'Fiber'}
        fiber_elements_by_uid = {uid: el for uid, el in elements_by_uid.items() if el.get('type') == 'Fiber'}
        
        # Construir mapa de conexiones
        connections_map = {}
        for conn in connections:
            from_n, to_n = conn['from_node'], conn['to_node']
            connections_map.setdefault(from_n, []).append(to_n)
        
        # Procesamiento de conexiones para manejar cadenas de fibras
        processed_connections = []
        processed_edge_tuples = set()
        
        def find_fiber_chain_end(start_fiber_uid, visited=None):
            """Encontrar el final de una cadena de fibras."""
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
        
        # Procesar conexiones desde nodos reales
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
        
        # Determinar tipo de gráfico basado en coordenadas
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


def _create_map_plot(nodes_to_plot, processed_connections, data):
    """Crea un gráfico basado en mapa para la topología de red."""
    fig = go.Figure()

    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(nodes_to_plot, edfa_equipment_data)
    
    nodes_by_uid = {node['uid']: node for node in nodes_to_plot}
    
    # Dibujar conexiones (líneas) entre nodos
    for conn in processed_connections:
        from_uid, to_uid = conn['from_node'], conn['to_node']
        if from_uid in nodes_by_uid and to_uid in nodes_by_uid:
            from_node, to_node = nodes_by_uid[from_uid], nodes_by_uid[to_uid]
            
            from_lat, from_lon = from_node['metadata']['latitude'], from_node['metadata']['longitude']
            to_lat, to_lon = to_node['metadata']['latitude'], to_node['metadata']['longitude']
            
            # Generar puntos intermedios para línea recta
            num_points = 20
            lats = [from_lat + (to_lat - from_lat) * i / (num_points - 1) for i in range(num_points)]
            lons = [from_lon + (to_lon - from_lon) * i / (num_points - 1) for i in range(num_points)]
            
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=lons,
                lat=lats,
                hoverinfo='none',
                line=dict(width=3, color='#FF0000'),
                showlegend=False
            ))

    # Agregar marcadores para tooltips de fibras
    fiber_hover_lons, fiber_hover_lats, fiber_hover_texts = [], [], []
    for conn in processed_connections:
        if conn.get('fiber_chain'):
            from_uid, to_uid = conn['from_node'], conn['to_node']
            if from_uid in nodes_by_uid and to_uid in nodes_by_uid:
                from_node, to_node = nodes_by_uid[from_uid], nodes_by_uid[to_uid]
                fiber_hover_lons.append((from_node['metadata']['longitude'] + to_node['metadata']['longitude']) / 2)
                fiber_hover_lats.append((from_node['metadata']['latitude'] + to_node['metadata']['latitude']) / 2)
                fiber_hover_texts.append(get_fiber_chain_tooltip_text(conn['fiber_chain'], edfa_equipment_data))
    
    if fiber_hover_texts:
        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=fiber_hover_lons,
            lat=fiber_hover_lats,
            marker=dict(size=25, color='rgba(0,0,0,0)'),
            hovertext=fiber_hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False,
            name="Fibers"
        ))

    # Separar nodos por tipo
    transceivers = []
    edfas = []
    other_nodes = []
    
    for i, node in enumerate(nodes_to_plot):
        node_info = {
            'lat': node['metadata']['latitude'],
            'lon': node['metadata']['longitude'],
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
    
    # Agregar transceivers
    if transceivers:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[t['lon'] for t in transceivers],
            lat=[t['lat'] for t in transceivers],
            text=[t['uid'] for t in transceivers],
            hovertext=[t['hover'] for t in transceivers],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(size=12, color=[t['color'] for t in transceivers], symbol='circle'),
            textposition='bottom right',
            textfont=dict(size=12, color='black'),
            showlegend=False,
            name="Transceivers"
        ))
    
    # Agregar EDFAs
    if edfas:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[e['lon'] for e in edfas],
            lat=[e['lat'] for e in edfas],
            text=[e['uid'] for e in edfas],
            hovertext=[e['hover'] for e in edfas],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(size=8, color='#FF0000', symbol='circle'),
            textposition='bottom right',
            textfont=dict(size=10, color='red', family='Arial'),
            showlegend=False,
            name="EDFAs"
        ))
    
    # Agregar otros nodos
    if other_nodes:
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=[o['lon'] for o in other_nodes],
            lat=[o['lat'] for o in other_nodes],
            text=[o['uid'] for o in other_nodes],
            hovertext=[o['hover'] for o in other_nodes],
            hovertemplate='%{hovertext}<extra></extra>',
            marker=dict(size=20, color=[o['color'] for o in other_nodes], symbol='circle'),
            textposition='bottom right',
            textfont=dict(size=11, color='black'),
            showlegend=False,
            name="Other Nodes"
        ))

    # Configurar diseño del mapa
    all_lats = [node['metadata']['latitude'] for node in nodes_to_plot]
    all_lons = [node['metadata']['longitude'] for node in nodes_to_plot]
    
    center_lat = np.mean(all_lats) if all_lats else 0
    center_lon = np.mean(all_lons) if all_lons else 0
    
    # Calcular zoom apropiado
    zoom_level = 5
    if len(all_lats) == 2:
        lat_diff = abs(max(all_lats) - min(all_lats))
        lon_diff = abs(max(all_lons) - min(all_lons))
        max_diff = max(lat_diff, lon_diff)
        
        if max_diff < 0.1:
            zoom_level = 12
        elif max_diff < 0.5:
            zoom_level = 10
        elif max_diff < 2:
            zoom_level = 8
        elif max_diff < 5:
            zoom_level = 6
        else:
            zoom_level = 4
    
    fig.update_layout(
        title_text=data.get('network_name', 'Topología de Red'),
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

    # Determinar el orden horizontal de nodos
    all_elements = data.get('elements', [])
    all_connections = data.get('connections', [])
    ordered_node_uids = find_path_for_layout(all_elements, all_connections)
    
    nodes_by_uid = {node['uid']: node for node in nodes_to_plot}
    
    # Filtrar y ordenar los nodos
    ordered_nodes_to_plot = [nodes_by_uid[uid] for uid in ordered_node_uids if uid in nodes_by_uid]
    
    if not ordered_nodes_to_plot:
        ordered_nodes_to_plot = sorted(nodes_to_plot, key=lambda x: x['uid'])
        ordered_node_uids = [node['uid'] for node in ordered_nodes_to_plot]

    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(ordered_nodes_to_plot, edfa_equipment_data)
    
    # Asignar coordenadas horizontales
    node_positions = {uid: {'x': i * 100, 'y': 100} for i, uid in enumerate(ordered_node_uids)}

    # Preparar listas para puntos de hover de fibras
    hover_x, hover_y, hover_texts = [], [], []

    # Dibujar conexiones
    for conn in processed_connections:
        from_uid, to_uid = conn['from_node'], conn['to_node']
        if from_uid in node_positions and to_uid in node_positions:
            from_pos, to_pos = node_positions[from_uid], node_positions[to_uid]

            fig.add_trace(go.Scatter(
                x=[from_pos['x'], to_pos['x']],
                y=[from_pos['y'], to_pos['y']],
                mode='lines', line=dict(width=2, color='gray'),
                hoverinfo='none', showlegend=False
            ))

            # Agregar marcador invisible para tooltip de fibra
            if conn.get('fiber_chain'):
                hover_x.append((from_pos['x'] + to_pos['x']) / 2)
                hover_y.append(from_pos['y'])
                hover_texts.append(get_fiber_chain_tooltip_text(conn['fiber_chain'], edfa_equipment_data))

    # Agregar puntos de hover invisibles
    if hover_texts:
        fig.add_trace(go.Scatter(
            x=hover_x, y=hover_y, mode='markers',
            marker=dict(size=25, color='rgba(0,0,0,0)'),
            hovertext=hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False
        ))

    # Agregar marcadores de nodos
    node_x = [node_positions[el['uid']]['x'] for el in ordered_nodes_to_plot]
    node_y = [node_positions[el['uid']]['y'] for el in ordered_nodes_to_plot]
    node_text = [el['uid'] for el in ordered_nodes_to_plot]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        text=node_text, 
        hovertext=node_hover_texts,
        hovertemplate='%{hovertext}<extra></extra>', 
        mode='markers+text',
        textposition='bottom center',
        marker=dict(size=20, color=node_colors, symbol=node_symbols),
        textfont=dict(size=11), 
        showlegend=False
    ))
    
    # Detectar si es topología punto a punto
    is_point_to_point = len(ordered_nodes_to_plot) == 2
    
    fig.update_layout(
        title_text=data.get('network_name', 'Topología de Red Punto a Punto' if is_point_to_point else 'Topología de Red'),
        showlegend=False, 
        xaxis=dict(visible=False, range=[-50, len(ordered_nodes_to_plot) * 100 - 50]),
        yaxis=dict(visible=False, range=[0, 200]),
        hovermode='closest', 
        plot_bgcolor='#f8f9fa',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


# =================== END TOPOLOGY VISUALIZATION FUNCTIONS ===================

# =================== WEB APPLICATION FUNCTIONS ===================

def enhance_elements_with_parameters(elements):
    """Enhance elements with parameter information for editing."""
    enhanced_elements = []
    
    # Identify source and destination transceivers
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    source_transceiver = None
    destination_transceiver = None
    
    if len(transceivers) >= 2:
        source_uid, dest_uid = identify_source_destination_transceivers(elements)
        source_transceiver = source_uid
        destination_transceiver = dest_uid
    
    for element in elements:
        enhanced_element = element.copy()
        element_type = element.get('type', '')
        
        if element_type == 'Transceiver':
            # Determine if this is source or destination transceiver
            if element.get('uid') == source_transceiver:
                enhanced_element['parameters'] = get_source_transceiver_defaults()
                enhanced_element['role'] = 'source'
            elif element.get('uid') == destination_transceiver:
                enhanced_element['parameters'] = get_destination_transceiver_defaults()
                enhanced_element['role'] = 'destination'
            else:
                # Fallback for other transceivers
                enhanced_element['parameters'] = get_source_transceiver_defaults()
                enhanced_element['role'] = 'source'
            
        elif element_type == 'Fiber':
            # Fiber parameters from existing params
            existing_params = element.get('params', {})
            enhanced_element['parameters'] = get_fiber_defaults(existing_params)
            
        elif element_type == 'Edfa':
            # Get EDFA parameters from equipment configuration
            type_variety = element.get('type_variety', 'std_medium_gain')
            edfa_config = find_edfa_config(type_variety)
            operational = element.get('operational', {})
            enhanced_element['parameters'] = get_edfa_defaults(edfa_config, operational)
        
        enhanced_elements.append(enhanced_element)
    
    return enhanced_elements


def get_source_transceiver_defaults():
    """Get default parameters for source transceivers (transmitters)."""
    return {
        'P_tot_dbm_input': {'value': 15.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Total del Transmisor (P_tot_dbm_input) - Potencia total de salida del transmisor que será dividida entre todos los canales'},
        'tx_osnr': {'value': 45.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - OSNR inicial del transmisor usado para los cálculos'}
    }


def get_destination_transceiver_defaults():
    """Get default parameters for destination transceivers (receivers)."""
    return {
        'sens': {'value': -25.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - Nivel mínimo de potencia que el receptor puede detectar correctamente'}
    }


def get_fiber_defaults(existing_params):
    """Get parameters for fiber elements."""
    return {
        'loss_coef': {'value': existing_params.get('loss_coef', 0.2), 'unit': 'dB/km', 'editable': False, 'tooltip': 'Coeficiente de Pérdida de Fibra - El coeficiente que representa la tasa de pérdida de la fibra'},
        'length_km': {'value': existing_params.get('length', 80), 'unit': 'km', 'editable': False, 'tooltip': 'Longitud de Fibra (km) - La longitud total de la sección de fibra en kilómetros'},
        'con_in': {'value': existing_params.get('con_in', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Entrada - El tipo de conector usado en la entrada de la fibra'},
        'con_out': {'value': existing_params.get('con_out', 0.5), 'unit': 'dB', 'editable': False, 'tooltip': 'Conector de Salida - El tipo de conector usado en la salida de la fibra'},
        'att_in': {'value': existing_params.get('att_in', 0.0), 'unit': 'dB', 'editable': False, 'tooltip': 'Pérdidas de Entrada - Pérdidas encontradas en el lado de entrada de la fibra'}
    }


def get_edfa_defaults(edfa_config, operational):
    """Get parameters for EDFA elements."""
    return {
        'gain_flatmax': {'value': edfa_config.get('gain_flatmax', 26), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Plana Máxima - La ganancia máxima alcanzada por el amplificador under condiciones planas'},
        'gain_min': {'value': edfa_config.get('gain_min', 15), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Mínima - La ganancia mínima alcanzable por el amplificador'},
        'p_max': {'value': edfa_config.get('p_max', 23), 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Máxima - La potencia de salida máxima proporcionada por el amplificador'},
        'nf0': {'value': edfa_config.get('nf_min', edfa_config.get('nf0', 5)), 'unit': 'dB', 'editable': True, 'tooltip': 'Factor de Ruido (NF) - La figura de ruido del amplificador que afecta la relación señal-ruido. Este valor puede ser modificado para coincidir con especificaciones del fabricante o condiciones operacionales específicas.'},
        'gain_target': {'value': operational.get('gain_target', 20), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Objetivo - La ganancia deseada a ser alcanzada por el amplificador basada en configuraciones operacionales'}
    }


def find_edfa_config(type_variety):
    """Find EDFA configuration by type_variety."""
    config = edfa_equipment_data.get(type_variety, {})
    if not config:
        # Return default values if not found
        return {'gain_flatmax': 26, 'gain_min': 15, 'p_max': 23, 'nf_min': 6}
    return config


def process_scenario02_data(file):
    """Process uploaded JSON file and return network visualization data."""
    if file.filename == '':
        return {'error': "No se seleccionó ningún archivo"}

    if not file.filename.endswith('.json'):
        return {'error': "Tipo de archivo inválido. Por favor, suba un archivo .json"}

    try:
        data = json.load(file.stream)
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        if not elements:
            return {'error': "No hay elementos en la topología"}
        
        # Enhance elements with parameters
        enhanced_elements = enhance_elements_with_parameters(elements)
        enhanced_data = data.copy()
        enhanced_data['elements'] = enhanced_elements

        # Create topology visualization
        topology_viz = create_topology_visualization_from_data(data)
        
        if 'error' in topology_viz:
            return {'error': topology_viz['error']}
        
        return {
            'graph_json': topology_viz['figure'].to_json(),
            'enhanced_data': enhanced_data
        }
        
    except Exception as e:
        return {'error': f"Error al procesar el archivo: {e}"}


def create_topology_visualization_from_data(data):
    """Create topology visualization from data dict instead of file path."""
    try:
        elements = data.get('elements', [])
        connections = data.get('connections', [])
        
        if not elements:
            return {'error': 'No hay elementos en la topología'}
        
        # Process elements for visualization (same logic as create_topology_visualization)
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


def update_scenario02_parameters():
    """Update network parameters for scenario02 elements."""
    try:
        data = request.get_json()
        element_uid = data.get('element_uid')
        parameter_name = data.get('parameter_name')
        new_value = data.get('new_value')
        
        # Here you would normally save the updated parameters to a database or session
        # For now, just return success
        return jsonify({
            'success': True,
            'message': f'Parameter {parameter_name} updated for element {element_uid}',
            'element_uid': element_uid,
            'parameter_name': parameter_name,
            'new_value': new_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def calculate_scenario02():
    """Calculate scenario02 network based on loaded topology and user parameters."""
    try:
        data = request.get_json()
        topology_data = data.get('topology_data', {})
        
        # Execute the calculation
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


# =================== END WEB APPLICATION FUNCTIONS ===================


def calculate_scenario02_network(calculation_params):
    """
    Calculate scenario02 network based on provided parameters.
    
    Args:
        calculation_params (dict): Dictionary containing topology_data and other parameters
        
    Returns:
        dict: Dictionary containing calculation results and plots
    """
    try:
        if not GNPY_AVAILABLE:
            return {
                'success': False,
                'error': 'gnpy library is not available. Please install gnpy to perform calculations.'
            }
            
        topology_data = calculation_params.get('topology_data', {})
        
        # Declare global variables at the top
        global sens, P_tot_dbm_input, tx_power_dbm
        
        # Extract parameters from calculation_params
        user_sens = calculation_params.get('sens', sens)
        user_P_tot_dbm = calculation_params.get('P_tot_dbm_input', P_tot_dbm_input)
        user_nf_values = calculation_params.get('nf_values', [])
        
        # Create a temporary topology file for gnpy
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(topology_data, temp_file)
            temp_topology_path = temp_file.name
        
        try:
            # Load the topology
            topology = load_topology(temp_topology_path)
            network = topology['network']
            transceivers = topology['transceivers']
            edfas = topology['edfas']
            fibers = topology['fibers']
            tx = topology['tx']
            rx = topology['rx']
            
            # Initialize plot data
            plot_data = initialize_plot_data()
            current_distance = 0
            
            # Use provided NF values or defaults
            if not user_nf_values:
                nf_values = [4.5] * len(edfas)  # Default NF values
            else:
                nf_values = user_nf_values
                
            # Update global parameters
            sens = user_sens
            P_tot_dbm_input = user_P_tot_dbm
            tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch)
            
            # Create spectral information
            freq = [f_min + spacing * i for i in range(nch)]
            signal = [dbm2watt(tx_power_dbm)] * nch
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
            
            # Set initial ASE
            lin_osnr0 = 10**(tx_osnr / 10)
            si.ase = np.array([np.sum(ch.power) / lin_osnr0 for ch in si.carriers], dtype=np.float64)
            
            # Process the network
            final_si, final_distance, p_rb, final_osnr, final_ase_lin = process_network_elements(
                si, edfas, fibers, tx, rx, nf_values, current_distance, plot_data
            )
            
            # Create plots
            plots = create_plotly_figures(plot_data)
            
            # Format results
            p_rb_formatted = f"{p_rb:.2f}" if p_rb != -0.00 else "0.00"
            sens_formatted = f"{sens:.2f}" if sens != -0.00 else "0.00"
            
            sensitivity_check = p_rb >= sens
            sensitivity_message = (
                f"¡Éxito! La potencia recibida ({p_rb_formatted} dBm) es mayor o igual que la sensibilidad ({sens_formatted} dBm)."
                if sensitivity_check else
                f"Advertencia: La potencia recibida ({p_rb_formatted} dBm) es menor que la sensibilidad ({sens_formatted} dBm)."
            )
            
            return {
                'success': True,
                'message': 'Calculation completed successfully',
                'results': {
                    'final_power_dbm': p_rb,
                    'final_osnr_db': final_osnr,
                    'sensitivity_check': sensitivity_check,
                    'sensitivity_message': sensitivity_message,
                    'final_distance_km': final_distance,
                    'nch': nch,
                    'tx_power_per_channel_dbm': tx_power_dbm,
                    'P_tot_dbm_input': P_tot_dbm_input
                },
                'plots': {
                    'signal_power_plot': plots['signal_power_plot'].to_json(),
                    'ase_power_plot': plots['ase_power_plot'].to_json(),
                    'osnr_plot': plots['osnr_plot'].to_json()
                }
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_topology_path)
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Calculation failed: {str(e)}'
        }


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


def load_topology(topology_file_path, equipment_file_path=("versionamientos/eqpt_config.json")):
    """
    Load network topology from JSON file and return network elements.
    
    Args:
        topology_file_path (str): Path to the topology JSON file
        equipment_file_path (str): Path to the equipment configuration file
        
    Returns:
        dict: Dictionary containing network elements:
            - 'network': The loaded network object
            - 'transceivers': List of transceivers sorted by uid
            - 'edfas': List of EDFAs sorted by uid
            - 'fibers': List of fibers sorted by uid
            - 'tx': First transceiver (transmitter)
            - 'rx': Last transceiver (receiver)
    """
    if not GNPY_AVAILABLE:
        raise ImportError("gnpy library is not available. Please install gnpy to load topologies.")
        
    try:
        # Use full path for equipment file
        if not os.path.isabs(equipment_file_path):
            equipment_file_path = os.path.join(os.path.dirname(__file__), equipment_file_path)
        
        # Load equipment and network
        equipment = load_equipment(Path(equipment_file_path))
        network = load_network(Path(topology_file_path), equipment)
        
        # Extract and sort network elements
        transceivers = sorted([n for n in network.nodes if isinstance(n, Transceiver)], key=lambda x: x.uid)
        edfas = sorted([n for n in network.nodes if isinstance(n, Edfa)], key=lambda x: x.uid)
        fibers = sorted([n for n in network.nodes if isinstance(n, Fiber)], key=lambda x: x.uid)
        
        # Validate topology
        if len(transceivers) < 2:
            raise ValueError(f"Topology must have at least 2 transceivers, found {len(transceivers)}")
        
        # Assign tx and rx (first and last transceivers)
        tx = transceivers[0]
        rx = transceivers[-1]
        
        return {
            'network': network,
            'transceivers': transceivers,
            'edfas': edfas,
            'fibers': fibers,
            'tx': tx,
            'rx': rx
        }
        
    except Exception as e:
        raise Exception(f"Error loading topology: {e}")

# 5) Default topology file (can be overridden by web interface)
default_topology_file = "topologiaEdfa3.json"

# --- Listas para almacenar datos para las gráficas ---
def initialize_plot_data():
    """Initialize plot data structure."""
    return {
        'distance': [],
        'signal_power': [],
        'ase_power': [], 
        'osnr_bw': []
    }

# Función para añadir un punto a las listas de plot
def add_plot_point(plot_data, dist, si_current, osnr_val):
    plot_data['distance'].append(dist)
    plot_data['signal_power'].append(watt2dbm(sum(ch.power[0] for ch in si_current.carriers)))
    # Sumamos la potencia de ruido ASE de todos los canales de GNPy
    plot_data['ase_power'].append(watt2dbm(sum(si_current.ase)))
    plot_data['osnr_bw'].append(osnr_val)

# 6) Core network processing functions
def process_network_elements(si, edfas, fibers, tx, rx, nf_values, current_distance, plot_data):
    """
    Process network elements dynamically regardless of their number.
    
    Args:
        si: Spectral information object
        edfas: List of EDFA elements
        fibers: List of fiber elements  
        tx: Transmitter element
        rx: Receiver element
        nf_values: List of noise figures for EDFAs
        current_distance: Current distance tracker
        plot_data: Plot data dictionary
    
    Returns:
        tuple: (final_si, final_distance, final_power_dbm, final_osnr_db, final_ase_lin)
    """
    if not GNPY_AVAILABLE:
        raise ImportError("gnpy library is not available. Please install gnpy to process network elements.")
    
    # Constants
    QUANTUM_NOISE_FLOOR_DBM = -58.0
    
    # Initialize tracking variables
    current_total_ase_lin_for_parallel_calc = dbm2watt(-150.0)  # Very small, negligible value
    
    # Process transmitter (starting point)
    p_tx = P_tot_dbm_input
    o_tx = tx_osnr
    add_plot_point(plot_data, current_distance, si, o_tx)
    
    # Process EDFAs and fibers in sequence
    # Typically: EDFA -> Fiber -> EDFA -> Fiber -> ... -> EDFA
    for i, edfa in enumerate(edfas):
        # Process EDFA
        si = edfa(si)
        p_edfa = watt2dbm(sum(ch.power[0] for ch in si.carriers))
        o_edfa = get_avg_osnr_db(si)
        
        # Calculate EDFA noise contribution
        gain_db = edfa.operational.gain_target
        if i < len(nf_values):
            noise_factor_db = nf_values[i]
        else:
            # Default NF if not provided
            noise_factor_db = 4.5
        
        p_ase_edfa_lin_manual = dbm2watt(QUANTUM_NOISE_FLOOR_DBM + noise_factor_db + gain_db)
        current_total_ase_lin_for_parallel_calc = (current_total_ase_lin_for_parallel_calc * 10**(gain_db / 10)) + p_ase_edfa_lin_manual
        
        add_plot_point(plot_data, current_distance, si, o_edfa)
        
        # Process corresponding fiber span (if exists)
        if i < len(fibers):
            fiber = fibers[i]
            
            # Store noise states before fiber
            ase_before_fiber = si.ase.copy()
            nli_before_fiber = si.nli.copy()
            
            # Set fiber input power
            p_in_fiber = watt2dbm(sum(ch.power[0] for ch in si.carriers))
            ase_in_fiber_lin_for_parallel_calc = current_total_ase_lin_for_parallel_calc
            
            fiber.ref_pch_in_dbm = p_in_fiber - 10 * np.log10(nch)
            si = fiber(si)
            
            # Calculate fiber loss
            loss_dB = fiber.params.loss_coef * fiber.params.length + fiber.params.con_in + fiber.params.con_out + fiber.params.att_in
            loss_lin = 10**(-loss_dB / 10)
            
            # Apply loss to noise components
            si.ase = ase_before_fiber * loss_lin
            si.nli = nli_before_fiber * loss_lin
            
            # Update ASE tracking
            current_total_ase_lin_for_parallel_calc = ase_in_fiber_lin_for_parallel_calc * loss_lin
            
            # Calculate fiber output
            p_fiber = watt2dbm(sum(ch.power[0] for ch in si.carriers))
            o_fiber = get_avg_osnr_db(si)
            
            # Update distance
            current_distance += fiber.params.length / 1000
            
            add_plot_point(plot_data, current_distance, si, o_fiber)
    
    # Process receiver (final point)
    si = rx(si)
    p_rx = watt2dbm(sum(ch.power[0] for ch in si.carriers))
    o_rx = get_avg_osnr_db(si)
    
    add_plot_point(plot_data, current_distance, si, o_rx)
    
    return si, current_distance, p_rx, o_rx, current_total_ase_lin_for_parallel_calc


def create_plotly_figures(plot_data):
    """Create Plotly figures from plot data."""
    
    # Plot 1 - Signal Power
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=plot_data['distance'],
        y=plot_data['signal_power'],
        mode='lines+markers',
        name='P_signal (dBm)',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig1.update_layout(
        title='Evolución de la Potencia de Señal a lo largo del enlace óptico',
        xaxis_title='Distancia (km)',
        yaxis_title='Potencia (dBm)',
        width=800,
        height=400,
        showlegend=True,
        grid=True
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Plot 2 - ASE Power
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=plot_data['distance'],
        y=plot_data['ase_power'],
        mode='lines+markers',
        name='P_ASE (dBm)',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    fig2.update_layout(
        title='Evolución de la Potencia de Ruido ASE a lo largo del enlace óptico',
        xaxis_title='Distancia (km)',
        yaxis_title='Potencia (dBm)',
        width=800,
        height=400,
        showlegend=True,
        grid=True
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Plot 3 - OSNR
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=plot_data['distance'],
        y=plot_data['osnr_bw'],
        mode='lines+markers',
        name='OSNR_bw (dB)',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    fig3.update_layout(
        title='Evolución de OSNR a lo largo del enlace óptico',
        xaxis_title='Distancia (km)',
        yaxis_title='OSNR (dB)',
        width=800,
        height=400,
        showlegend=True,
        grid=True
    )
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return {
        'signal_power_plot': fig1,
        'ase_power_plot': fig2,
        'osnr_plot': fig3
    }


# =================== ADDITIONAL TOPOLOGY FEATURES ===================

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
