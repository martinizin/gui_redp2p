import plotly.graph_objects as go
import json
import os
import numpy as np
from flask import jsonify, request

# 1) PARÁMETROS
f_min, f_max = 191.3e12, 195.1e12
spacing = 50e9
roll_off = 0.15
tx_osnr = 45  # dB inicial, este valor es para los cálculos de gnpy de OSNR_bw
baud_rate = 32e9
B_n=12.5e9 #no sea modificable

# Import gnpy modules for proper OSNR_bw calculation
try:
    from pathlib import Path
    from gnpy.tools.json_io import load_equipment, load_network
    from gnpy.core.info import create_arbitrary_spectral_information
    from gnpy.core.utils import dbm2watt as gnpy_dbm2watt, watt2dbm as gnpy_watt2dbm, lin2db as gnpy_lin2db
    from gnpy.core.elements import Transceiver, Fiber, Edfa
    GNPY_AVAILABLE = True
except ImportError:
    print("Warning: gnpy not available. OSNR_bw calculations may not be accurate.")
    GNPY_AVAILABLE = False


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
            Encontrar recursivamente el final de una cadena de fibras.
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
            has_coordinates = all(
                isinstance(node.get('metadata'), dict) and
                'latitude' in node['metadata'] and
                'longitude' in node['metadata'] and
                isinstance(node['metadata']['latitude'], (int, float)) and
                isinstance(node['metadata']['longitude'], (int, float)) and
                not (node['metadata']['latitude'] == 0 and node['metadata']['longitude'] == 0)  # Evitar coordenadas (0,0) que pueden ser placeholders
                for node in plot_nodes
            )

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
    
    # Dibujar conexiones (líneas) entre nodos - forzar líneas rectas
    for conn in processed_connections:
        from_uid, to_uid = conn['from_node'], conn['to_node']
        if from_uid in nodes_by_uid and to_uid in nodes_by_uid:
            from_node, to_node = nodes_by_uid[from_uid], nodes_by_uid[to_uid]
            
            # Crear líneas rectas agregando puntos intermedios para evitar curvatura
            from_lat, from_lon = from_node['metadata']['latitude'], from_node['metadata']['longitude']
            to_lat, to_lon = to_node['metadata']['latitude'], to_node['metadata']['longitude']
            
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
                fiber_hover_lons.append((from_node['metadata']['longitude'] + to_node['metadata']['longitude']) / 2)
                fiber_hover_lats.append((from_node['metadata']['latitude'] + to_node['metadata']['latitude']) / 2)
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
            textposition='bottom right',
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
        if 'metadata' in node and 'latitude' in node['metadata'] and 'longitude' in node['metadata']:
            all_lats.append(node['metadata']['latitude'])
            all_lons.append(node['metadata']['longitude'])
    
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
    
    # Asignar coordenadas horizontales
    node_positions = {uid: {'x': i * 100, 'y': 100} for i, uid in enumerate(ordered_node_uids)}

    # Preparar listas para puntos de hover de span de fibra
    hover_x, hover_y, d2_hover_texts = [], [], []

    # Dibujar conexiones (líneas) entre nodos
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
    node_x = [node_positions[el['uid']]['x'] for el in ordered_nodes_to_plot]
    node_y = [node_positions[el['uid']]['y'] for el in ordered_nodes_to_plot]
    node_text_on_graph = [el['uid'] for el in ordered_nodes_to_plot]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        text=node_text_on_graph, 
        hovertext=node_hover_texts,
        hovertemplate='%{hovertext}<extra></extra>', 
        mode='markers+text',
        textposition='bottom center',
        marker=dict(size=20, color=node_colors, symbol=node_symbols),
        textfont=dict(size=11), 
        showlegend=False
    ))
    
    # Detectar si es topología punto a punto para ajustar título
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

def get_source_transceiver_defaults():
    """Obtener parámetros por defecto para transceptores de origen (transmisores)."""
    return {
        'P_tot_dbm_input': {'value': 0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Total del Transmisor (P_tot_dbm_input) - Potencia total de salida del transmisor que será dividida entre todos los canales'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - OSNR inicial del transmisor usado para los cálculos'}
    }

def get_destination_transceiver_defaults():
    """Obtener parámetros por defecto para transceptores de destino (receptores)."""
    return {
        'sens': {'value': -30.0, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - Nivel mínimo de potencia que el receptor puede detectar correctamente'}
    }

def get_transceiver_defaults():
    """Obtener parámetros por defecto para transceptores (función heredada para compatibilidad)."""
    return {
        'p_rb': {'value': -17.86, 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia de Señal Recibida - Modifique este valor para ajustar la potencia de señal'},
        'tx_osnr': {'value': 40.0, 'unit': 'dB', 'editable': True, 'tooltip': 'OSNR de Transmisión - Modifique el valor OSNR para optimizar la calidad de señal'},
        'sens': {'value': 1, 'unit': 'dBm', 'editable': True, 'tooltip': 'Sensibilidad del Receptor - El nivel de sensibilidad del receptor a las señales entrantes'}
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
    """Obtener parámetros para elementos EDFA."""
    return {
        'gain_flatmax': {'value': edfa_config.get('gain_flatmax', 26), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Plana Máxima - La ganancia máxima alcanzada por el amplificador under condiciones planas'},
        'gain_min': {'value': edfa_config.get('gain_min', 15), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Mínima - La ganancia mínima alcanzable por el amplificador'},
        'p_max': {'value': edfa_config.get('p_max', 23), 'unit': 'dBm', 'editable': True, 'tooltip': 'Potencia Máxima - La potencia de salida máxima proporcionada por el amplificador'},
        'nf0': {'value': edfa_config.get('nf_min', edfa_config.get('nf0', 5)), 'unit': 'dB', 'editable': True, 'tooltip': 'Factor de Ruido (NF) - La figura de ruido del amplificador que afecta la relación señal-ruido. Este valor puede ser modificado para coincidir con especificaciones del fabricante o condiciones operacionales específicas.'},
        'gain_target': {'value': operational.get('gain_target', 20), 'unit': 'dB', 'editable': True, 'tooltip': 'Ganancia Objetivo - La ganancia deseada a ser alcanzada por el amplificador basada en configuraciones operacionales'}
    }

def find_edfa_config(type_variety):
    """Encontrar configuración EDFA por type_variety."""
    config = edfa_equipment_data.get(type_variety, {})
    if not config:
        # Devolver valores por defecto si no se encuentra
        return {'gain_flatmax': 26, 'gain_min': 15, 'p_max': 23, 'nf_min': 6}
    return config

def update_scenario02_parameters():
    """Actualizar parámetros de red para elementos de scenario02."""
    try:
        data = request.get_json()
        element_uid = data.get('element_uid')
        parameter_name = data.get('parameter_name')
        new_value = data.get('new_value')
        
        # Aquí normalmente guardarías los parámetros actualizados en una base de datos o sesión
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

def calculate_scenario02():
    """Calcular red scenario02 basado en topología cargada y parámetros del usuario."""
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

# Funciones de cálculo del notebook
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
    if lin <= 0:
        return -float('inf')
    return 10 * np.log10(lin)

def db2lin(db):
    """Convertir dB a lineal"""
    return 10 ** (db / 10)

def format_osnr(v):
    """Formatear valor OSNR para visualización"""
    return "∞" if np.isinf(v) else f"{v:.2f}"



def ensure_json_serializable(obj):
    """Convertir objetos numpy a tipos nativos de Python para serialización JSON"""
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
    Crear información espectral usando gnpy como en el notebook.
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
    Calcular OSNR promedio exactamente como en el notebook usando SpectralInformation de gnpy.
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

def get_avg_osnr_db(signal_power_lin_per_channel, ase_noise_lin_per_channel, nli_noise_lin_per_channel=0.0):
    total_noise_lin = ase_noise_lin_per_channel + nli_noise_lin_per_channel
    if total_noise_lin <= 0:
        return float('inf')
    osnr_lin = signal_power_lin_per_channel / total_noise_lin
    return lin2db(osnr_lin)


def classical_osnr_parallel(signal_power_dbm, ase_noise_dbm):
    if ase_noise_dbm == -float('inf') or ase_noise_dbm <= -190: # Considerar ruido muy bajo
        return float('inf') # OSNR muy alto si el ruido es casi cero
    
    signal_power_lin = dbm2watt(signal_power_dbm)
    ase_noise_lin = dbm2watt(ase_noise_dbm)
    
    if ase_noise_lin <= 0:
        return float('inf') # Evitar división por cero o negativo
    
    return lin2db(signal_power_lin / ase_noise_lin)

def test_gnpy_integration():
    """Función de prueba para verificar que la integración de gnpy funciona correctamente"""
    try:
        if not GNPY_AVAILABLE:
            return ensure_json_serializable({'status': 'error', 'message': 'gnpy not available'})
        
        # Parámetros de prueba (como en el notebook)
        f_min, f_max = 191.3e12, 195.1e12
        spacing = 50e9
        roll_off = 0.15
        tx_osnr = 45
        baud_rate = 32e9
        P_tot_dbm_input = 50.0
        
        nch = int(np.floor((f_max - f_min) / spacing)) + 1
        tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch)
        
        # Crear información espectral de prueba
        test_si = create_spectral_information_for_calculation(
            nch, tx_power_dbm, tx_osnr, f_min, spacing, baud_rate, roll_off
        )
        
        if test_si is None:
            return ensure_json_serializable({'status': 'error', 'message': 'Could not create spectral information'})
        
        # Calcular OSNR usando el método de gnpy
        osnr_bw = get_avg_osnr_db_from_spectral_info(test_si)
        
        return ensure_json_serializable({
            'status': 'success', 
            'message': f'gnpy integration working - OSNR_bw: {osnr_bw:.2f} dB',
            'nch': int(nch),
            'tx_power_dbm': float(tx_power_dbm),
            'osnr_bw': float(osnr_bw)
        })
        
    except Exception as e:
        return ensure_json_serializable({'status': 'error', 'message': f'gnpy integration test failed: {str(e)}'})

def validate_topology_requirements(elements, connections):
    """
    Validar que la topología cumpla con los requisitos mínimos:
    - Al menos 2 transceptores
    - Para topologías punto a punto (exactamente 2 transceivers): permite conexión directa
    - Para topologías complejas (más de 2 transceivers): requiere al menos 1 EDFA y 1 span de Fibra
    """
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    edfas = [e for e in elements if e.get('type') == 'Edfa']
    fibers = [e for e in elements if e.get('type') == 'Fiber']
    
    errors = []
    if len(transceivers) < 2:
        errors.append(f"Se requieren al menos 2 transceivers, encontrados: {len(transceivers)}")
        return errors
    
    # Verificar si es topología punto a punto (exactamente 2 transceivers)
    is_point_to_point = len(transceivers) == 2
    
    if is_point_to_point:
        # Para punto a punto, permitir conexión directa o con elementos intermedios
        # No requerir obligatoriamente EDFAs o Fibras
        pass
    else:
        # Para topologías complejas, mantener requisitos originales
        if len(edfas) < 1:
            errors.append(f"Para topologías complejas se requiere al menos 1 EDFA, encontrados: {len(edfas)}")
        if len(fibers) < 1:
            errors.append(f"Para topologías complejas se requiere al menos 1 span de fibra, encontrados: {len(fibers)}")
    
    return errors

def order_elements_by_topology(elements, connections):
    """
    Ordenar elementos de red basado en las conexiones de topología reales.
    Devuelve lista ordenada de elementos desde origen hasta destino.
    """
    # Validar requisitos mínimos
    validation_errors = validate_topology_requirements(elements, connections)
    if validation_errors:
        raise ValueError("Topología no válida: " + "; ".join(validation_errors))
    
    # Construir grafo de topología
    graph, elements_by_uid = build_topology_graph(elements, connections)
    
    # Identificar transceptores de origen y destino
    transceivers = [e for e in elements if e.get('type') == 'Transceiver']
    source_transceiver = None
    destination_transceiver = None
    
    # Encontrar transceptores con roles
    for t in transceivers:
        if t.get('role') == 'source':
            source_transceiver = t
        elif t.get('role') == 'destination':
            destination_transceiver = t
    
    # Identificación de respaldo si los roles no están establecidos
    if not source_transceiver or not destination_transceiver:
        source_uid, dest_uid = identify_source_destination_transceivers(elements)
        source_transceiver = elements_by_uid.get(source_uid)
        destination_transceiver = elements_by_uid.get(dest_uid)
    
    if not source_transceiver or not destination_transceiver:
        raise ValueError("No se pudieron identificar los transceivers de origen y destino")
    
    # Encontrar ruta a través de la red
    path = find_network_path(graph, source_transceiver['uid'], destination_transceiver['uid'])
    
    if not path:
        raise ValueError(f"No se encontró una ruta válida entre {source_transceiver['uid']} y {destination_transceiver['uid']}")
    
    # Convertir ruta a elementos ordenados
    ordered_elements = []
    for uid in path:
        if uid in elements_by_uid:
            ordered_elements.append(elements_by_uid[uid])
    
    return ordered_elements, source_transceiver, destination_transceiver

def calculate_scenario02_network(params):
    """
    Función de cálculo principal basada en la lógica del notebook.
    Espera params con:
    - topology_data: Datos de topología mejorados con elementos
    """
    try:
        # Extraer parámetros
        topology_data = params.get('topology_data', {})
        
        # Valores por defecto del notebook - coincidiendo exactamente
        f_min, f_max = 191.3e12, 195.1e12
        spacing = 50e9
        roll_off = 0.15
        baud_rate = 32e9
        B_n = 12.5e9  # Ancho de banda de referencia
        QUANTUM_NOISE_FLOOR_DBM = -58.0
        
        # Obtener topología de red
        elements = topology_data.get('elements', [])
        connections = topology_data.get('connections', [])
        
        # Ordenar elementos siguiendo las conexiones de topología reales
        try:
            ordered_elements, source_transceiver, destination_transceiver = order_elements_by_topology(elements, connections)
        except ValueError as e:
            return {'success': False, 'error': str(e)}

        # Extraer parámetros del diccionario 'parameters' de los transceptores identificados
        source_params = source_transceiver.get('parameters', {})
        dest_params = destination_transceiver.get('parameters', {})

        tx_osnr = source_params.get('tx_osnr', {}).get('value', 45.0)  # Usar 45 como en el notebook
        P_tot_dbm_input = source_params.get('P_tot_dbm_input', {}).get('value', 50.0)
        sens = dest_params.get('sens', {}).get('value', 20.0)
        
        # Calcular número de canales y potencia por canal (exactamente como en notebook)
        nch = int(np.floor((f_max - f_min) / spacing)) + 1
        tx_power_dbm = P_tot_dbm_input - 10 * np.log10(nch)  # Potencia por canal
        
        # Crear información espectral usando gnpy (exactamente como en el notebook)
        reference_si = create_spectral_information_for_calculation(
            nch, tx_power_dbm, tx_osnr, f_min, spacing, baud_rate, roll_off
        )
        
        # Extraer datos iniciales de la información espectral de referencia
        if GNPY_AVAILABLE and reference_si is not None:
            # Extraer potencia por canal inicial
            signal_power_per_channel = gnpy_dbm2watt(tx_power_dbm)  # Potencia por canal en watts
            # Extraer ASE inicial (relacionado con tx_osnr)
            lin_osnr0 = 10**(tx_osnr / 10)
            ase_power_per_channel = signal_power_per_channel / lin_osnr0
            # NLI inicial es 0
            nli_power_per_channel = 0.0
            
            print(f"Inicialización gnpy:")
            print(f"  Canales: {nch}")
            print(f"  Potencia por canal: {gnpy_watt2dbm(signal_power_per_channel):.2f} dBm")
            print(f"  ASE inicial por canal: {gnpy_watt2dbm(ase_power_per_channel):.2f} dBm")
            print(f"  OSNR inicial: {tx_osnr:.2f} dB")
        else:
            # Fallback si gnpy no está disponible
            signal_power_per_channel = dbm2watt(tx_power_dbm)
            ase_power_per_channel = signal_power_per_channel / (10**(tx_osnr / 10))
            nli_power_per_channel = 0.0
        
        # Variables para seguimiento paralelo de ASE (para compatibilidad con cálculos manuales)
        current_total_ase_lin_for_parallel_calc = dbm2watt(-150.0)  # Valor inicial muy pequeño para cálculo paralelo
        current_distance = 0.0
        current_power_dbm = P_tot_dbm_input  # Mostrar potencia total en transmisor
        
        # Almacenamiento de resultados
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
        
        def add_stage_result(name, distance, power_dbm, osnr_bw, osnr_01nm, osnr_parallel, ase_power_lin):
            """Agregar resultado de una etapa a los resultados"""            
            results['stages'].append({
                'name': str(name),
                'distance': float(distance),
                'power_dbm': float(power_dbm),
                'osnr_bw': format_osnr(float(osnr_bw)),
                'osnr_01nm': format_osnr(float(osnr_01nm)),
                'osnr_parallel': format_osnr(float(osnr_parallel)) if name == destination_transceiver.get('uid') else ''
            })
            
            # Obtener potencia ASE usando nuestras variables rastreadas
            ase_power_dbm = float((gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(ase_power_per_channel * nch)) if ase_power_per_channel > 0 else -150.0
            
            results['plot_data']['distance'].append(float(distance))
            results['plot_data']['signal_power'].append(float(power_dbm))
            results['plot_data']['ase_power'].append(float(ase_power_dbm))
            results['plot_data']['osnr_bw'].append(float(osnr_bw) if not np.isinf(osnr_bw) else 60.0)  # Límite para graficado
        
        # Función auxiliar para calcular OSNR usando enfoque de información espectral (coincidiendo con notebook)
        def calculate_current_osnr_bw():
            """Calcular OSNR_bw usando nuestras variables rastreadas como get_avg_osnr_db(si) del notebook"""
            try:
                # Calcular ruido total (ASE + NLI)
                total_noise_per_channel = ase_power_per_channel + nli_power_per_channel
                
                if total_noise_per_channel <= 0:
                    return float('inf')
                
                # Calcular OSNR por canal como en el notebook: sig / noise
                osnr_lin = signal_power_per_channel / total_noise_per_channel
                
                # Convertir a dB usando gnpy si está disponible, sino usar nuestra función
                if GNPY_AVAILABLE:
                    osnr_db = float(gnpy_lin2db(osnr_lin))
                else:
                    osnr_db = float(lin2db(osnr_lin))
                
                return osnr_db
            except Exception as e:
                print(f"Error in calculate_current_osnr_bw: {e}")
                print(f"  signal_power_per_channel: {signal_power_per_channel}")
                print(f"  ase_power_per_channel: {ase_power_per_channel}")
                print(f"  nli_power_per_channel: {nli_power_per_channel}")
                return float('inf')
        
        # Detectar si es topología punto a punto (solo 2 transceivers)
        transceivers_in_path = [el for el in ordered_elements if el.get('type') == 'Transceiver']
        is_point_to_point_direct = len(transceivers_in_path) == 2 and len(ordered_elements) == 2
        
        # Procesar cada elemento en la ruta ordenada
        for i, element in enumerate(ordered_elements):
            element_type = element.get('type', '')
            element_uid = element.get('uid', f'Element_{i}')
            params = element.get('parameters', {}) # Obtener parámetros editables
            
            if element_type == 'Transceiver':
                # Procesamiento de transceptor (parámetros ya extraídos)
                if element.get('uid') == source_transceiver.get('uid'):
                    # Transceptor de origen (transmisor) - siguiendo la lógica del notebook exactamente
                    # Site_A: p0 = P_tot_dbm_input, o0 = tx_osnr
                    current_osnr_bw = calculate_current_osnr_bw()  # Debería igualar tx_osnr debido a la inicialización
                    
                    # Debug: verificar OSNR inicial
                    print(f"Transceiver {element_uid}: tx_osnr={tx_osnr}, calculado={current_osnr_bw:.2f}")
                    
                    osnr_01nm_initial = current_osnr_bw + 10 * np.log10(baud_rate / B_n)
                    osnr_parallel_initial = classical_osnr_parallel(current_power_dbm, watt2dbm(current_total_ase_lin_for_parallel_calc))
                    
                    add_stage_result(element_uid, current_distance, current_power_dbm, current_osnr_bw, 
                                    osnr_01nm_initial, osnr_parallel_initial, current_total_ase_lin_for_parallel_calc)
                
                elif element.get('uid') == destination_transceiver.get('uid'):
                    # Transceptor de destino (receptor) - etapa final, no procesamiento, solo registrar
                    # Para conexión directa punto a punto, aplicar pérdida mínima de conexión
                    if is_point_to_point_direct:
                        # Aplicar pérdida mínima de conexión directa (conectores, etc.)
                        connection_loss_db = 1.0  # Pérdida típica de conectores
                        connection_loss_lin = 10**(-connection_loss_db / 10)
                        
                        # Aplicar pérdida a nuestras variables rastreadas
                        signal_power_per_channel *= connection_loss_lin
                        ase_power_per_channel *= connection_loss_lin
                        nli_power_per_channel *= connection_loss_lin
                        current_power_dbm -= connection_loss_db
                        current_total_ase_lin_for_parallel_calc *= connection_loss_lin
                    
                    # Usar el OSNR actual calculado desde la última etapa
                    current_osnr_bw = calculate_current_osnr_bw()
                    final_osnr_01nm = current_osnr_bw + 10 * np.log10(baud_rate / B_n)
                    final_osnr_parallel = classical_osnr_parallel(current_power_dbm, watt2dbm(current_total_ase_lin_for_parallel_calc))
                    
                    add_stage_result(element_uid, current_distance, current_power_dbm, current_osnr_bw, 
                                    final_osnr_01nm, final_osnr_parallel, current_total_ase_lin_for_parallel_calc)
                
            elif element_type == 'Edfa':
                # Procesamiento EDFA siguiendo exactamente la lógica del notebook
                gain_db = params.get('gain_target', {}).get('value', 17.0)
                noise_factor_db = params.get('nf0', {}).get('value', 6.0)  # Usar NF real de parámetros
                gain_lin = 10**(gain_db / 10)
                
                # Guardar valores antes del EDFA para debug
                signal_before = signal_power_per_channel
                ase_before = ase_power_per_channel
                nli_before = nli_power_per_channel
                
                # Aplicar ganancia a la señal (como notebook: si = edfa(si))
                signal_power_per_channel = signal_power_per_channel * gain_lin
                
                # Amplificar ASE existente (como en notebook)
                ase_power_per_channel = ase_power_per_channel * gain_lin
                
                # Agregar nuevo ASE de este EDFA usando la misma fórmula del notebook
                new_ase_total_lin = gnpy_dbm2watt(QUANTUM_NOISE_FLOOR_DBM + noise_factor_db + gain_db) if GNPY_AVAILABLE else dbm2watt(QUANTUM_NOISE_FLOOR_DBM + noise_factor_db + gain_db)
                new_ase_per_channel = new_ase_total_lin / nch
                ase_power_per_channel = ase_power_per_channel + new_ase_per_channel
                
                # Amplificar NLI existente (como en notebook)
                nli_power_per_channel = nli_power_per_channel * gain_lin
                
                # Actualizar potencia de visualización
                current_power_dbm = float((gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_power_per_channel * nch))
                
                # Debug: imprimir información detallada
                print(f"EDFA {element_uid}: gain={gain_db}dB, NF={noise_factor_db}dB")
                print(f"  Señal antes: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_before):.2f} dBm/ch")
                print(f"  Señal después: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_power_per_channel):.2f} dBm/ch")
                print(f"  ASE antes: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(ase_before):.2f} dBm/ch")
                print(f"  ASE después amp: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(ase_before * gain_lin):.2f} dBm/ch")
                print(f"  Nuevo ASE: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(new_ase_per_channel):.2f} dBm/ch")
                print(f"  ASE total: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(ase_power_per_channel):.2f} dBm/ch")
                
                # Cálculo manual de ASE para OSNR paralelo (seguimiento separado, como en notebook)
                p_ase_edfa_lin_manual = dbm2watt(QUANTUM_NOISE_FLOOR_DBM + noise_factor_db + gain_db)
                current_total_ase_lin_for_parallel_calc = (current_total_ase_lin_for_parallel_calc * gain_lin) + p_ase_edfa_lin_manual
                
                # Calcular OSNR_bw usando nuestras variables rastreadas (como notebook: o1 = get_avg_osnr_db(si))
                current_osnr_bw = calculate_current_osnr_bw()
                
                # Debug: mostrar OSNR calculado
                print(f"  OSNR_bw calculado: {current_osnr_bw:.2f} dB")
                
                osnr_01nm = current_osnr_bw + 10 * np.log10(baud_rate / B_n) if not np.isinf(current_osnr_bw) else float('inf')
                osnr_parallel = classical_osnr_parallel(current_power_dbm, watt2dbm(current_total_ase_lin_for_parallel_calc))
                
                add_stage_result(element_uid, current_distance, current_power_dbm, current_osnr_bw, 
                               osnr_01nm, osnr_parallel, current_total_ase_lin_for_parallel_calc)
                
            elif element_type == 'Fiber':
                # Procesamiento de fibra siguiendo exactamente la lógica del notebook
                length_km = params.get('length_km', {}).get('value', 80.0)
                loss_coef = params.get('loss_coef', {}).get('value', 0.2)
                con_in = params.get('con_in', {}).get('value', 0.5)
                con_out = params.get('con_out', {}).get('value', 0.5)
                att_in = params.get('att_in', {}).get('value', 0.0)
                
                # Calcular pérdida total (exactamente como notebook)
                total_loss_db = loss_coef * length_km + con_in + con_out + att_in
                loss_lin = 10**(-total_loss_db / 10)
                
                # Guardar valores antes del span para debug
                signal_before = signal_power_per_channel
                ase_before = ase_power_per_channel
                nli_before = nli_power_per_channel
                
                # Aplicar pérdida a la señal (como notebook: si = span(si))
                signal_power_per_channel = signal_power_per_channel * loss_lin
                
                # Aplicar pérdida a ASE y NLI (exactamente como en notebook)
                ase_power_per_channel = ase_before * loss_lin
                nli_power_per_channel = nli_before * loss_lin
                
                # Actualizar potencia de visualización
                current_power_dbm = float((gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_power_per_channel * nch))
                
                # Aplicar pérdida al seguimiento de ASE de cálculo paralelo
                current_total_ase_lin_for_parallel_calc *= loss_lin
                
                # Actualizar distancia (como notebook: current_distance += span.params.length / 1000)
                current_distance += length_km
                
                # Calcular OSNR_bw usando nuestras variables rastreadas (como notebook: o_s1 = get_avg_osnr_db(si))
                current_osnr_bw = calculate_current_osnr_bw()
                
                # Debug: mostrar información de fibra
                print(f"Fiber {element_uid}: loss={total_loss_db:.2f}dB, length={length_km}km")
                print(f"  Señal antes: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_before):.2f} dBm/ch")
                print(f"  Señal después: {(gnpy_watt2dbm if GNPY_AVAILABLE else watt2dbm)(signal_power_per_channel):.2f} dBm/ch")
                print(f"  OSNR_bw después de fibra: {current_osnr_bw:.2f} dB")
                    
                osnr_01nm = current_osnr_bw + 10 * np.log10(baud_rate / B_n) if not np.isinf(current_osnr_bw) else float('inf')
                osnr_parallel = classical_osnr_parallel(current_power_dbm, watt2dbm(current_total_ase_lin_for_parallel_calc))
                
                add_stage_result(element_uid, current_distance, current_power_dbm, current_osnr_bw, 
                               osnr_01nm, osnr_parallel, current_total_ase_lin_for_parallel_calc)
        
        # Resultados finales - coincidiendo con la lógica del notebook
        final_power_dbm = float(current_power_dbm)
            
        link_successful = final_power_dbm >= sens
        
        # Calcular valores OSNR finales usando información espectral (mismo método que el bucle de cálculo)
        final_osnr_bw = calculate_current_osnr_bw()
        final_osnr_01nm = final_osnr_bw + 10 * np.log10(baud_rate / B_n) if not np.isinf(final_osnr_bw) else float('inf')
        
        results['final_results'] = {
            'final_power_dbm': float(final_power_dbm),
            'receiver_sensitivity_dbm': float(sens),
            'link_successful': bool(link_successful),
            'power_margin_db': float(final_power_dbm - sens),
            'final_osnr_bw': format_osnr(float(final_osnr_bw)),
            'final_osnr_01nm': format_osnr(float(final_osnr_01nm)),
            'total_distance_km': float(current_distance),
            'nch': int(nch),
            'tx_power_per_channel_dbm': float(tx_power_dbm),
            'message': f"{'¡Éxito!' if link_successful else 'Advertencia:'} La potencia de la señal recibida ({final_power_dbm:.2f} dBm) es {'mayor o igual que' if link_successful else 'menor que'} la sensibilidad del receptor ({sens:.2f} dBm)."
        }
        
        # Generar gráficos
        results['plots'] = generate_scenario02_plots(results['plot_data'])
        
        # Asegurar que todos los resultados sean serializables en JSON
        return ensure_json_serializable(results)
        
    except Exception as e:
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
    
    # Agregar padding más generoso (15%) para evitar que las líneas estén muy cerca de los ejes
    signal_range = signal_max - signal_min
    signal_padding = max(signal_range * 0.15, 2.0)  # Mínimo 2 dB de padding
    
    ase_range = ase_max - ase_min
    ase_padding = max(ase_range * 0.15, 2.0)  # Mínimo 2 dB de padding
    
    osnr_range = osnr_max - osnr_min
    osnr_padding = max(osnr_range * 0.15, 1.0)  # Mínimo 1 dB de padding
    
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