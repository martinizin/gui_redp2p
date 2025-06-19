from flask import Flask, render_template, request, jsonify
from red import calcular_red, obtener_topologia_datos
import plotly.graph_objects as go
import json
import os

app = Flask(__name__)

# Load equipment configuration
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
    """Generates a formatted HTML string for an element's tooltip."""
    uid = element.get('uid', 'N/A')
    element_type = element.get('type', 'N/A')
    type_variety = element.get('type_variety', 'N/A')

    tooltip_html = (f"<b>uid:</b> {uid}<br>"
                    f"<b>type:</b> {element_type}<br>"
                    f"<b>type_variety:</b> {type_variety}<br>")

    # Add specific operational parameters for EDFAs
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

    # Add all 'params' for any element that has them (e.g., Fiber)
    if 'params' in element:
        params = element.get('params', {})
        tooltip_html += "<b>params:</b><br>"
        for k, v in params.items():
            tooltip_html += f"&nbsp;&nbsp;{k}: {v}<br>"
    
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

def dbm2mw(dbm):
    """Convierte dBm a mW."""
    return 10 ** (dbm / 10)

@app.route('/', methods=['GET'])
def index():
    # Provide initial data for the network and graphs
    nodos, enlaces = obtener_topologia_datos()

    
    initial_fiber_params = []
   
    default_span_length = 5 # Default length for initial calculation

    initial_params_for_calc = {
        'tx_power_dbm': 30, # Default from your JS
        'sensitivity_receiver_dbm': 14, # Default from your JS
        'fiber_params': [{ # A default single span
            'loss_coef': 0.2,
            'att_in': 0,
            'con_in': 0.25,
            'con_out': 0.30,
            'length_stretch': 23 # Default length from your JS form
        }]
    }
    # If you have multiple default spans from `obtener_topologia_datos` or JS, build `fiber_params` accordingly.

    resultados = calcular_red(initial_params_for_calc) # This now includes Plotly dicts

    return render_template('index.html',
                           resultados=resultados, # Contains Plotly graph data
                           nodos=nodos,
                           enlaces=enlaces,
                           initial_graph_dbm=resultados.get('plot_dbm_plotly'),
                           initial_graph_linear=resultados.get('plot_linear_plotly'))

@app.route('/scenario02', methods=['GET', 'POST'])
def scenario02():
    """Renders the scenario 2 page and handles file upload for network visualization."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('scenario2.html', error="No se encontró el archivo")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('scenario2.html', error="No se seleccionó ningún archivo")

        if file and file.filename.endswith('.json'):
            try:
                data = json.load(file.stream)
                elements = data.get('elements', [])
                connections = data.get('connections', [])
                fig = go.Figure()

                elements_by_uid = {el['uid']: el for el in elements}

                real_node_uids = {uid for uid, el in elements_by_uid.items() if el.get('type') != 'Fiber'}
                fiber_elements_by_uid = {uid: el for uid, el in elements_by_uid.items() if el.get('type') == 'Fiber'}

                connections_map = {}
                for conn in connections:
                    from_n, to_n = conn['from_node'], conn['to_node']
                    connections_map.setdefault(from_n, []).append(to_n)
                
                processed_connections = []
                processed_edge_tuples = set()
                for from_uid in real_node_uids:
                    if from_uid not in connections_map: continue
                    for target_uid in connections_map[from_uid]:
                        if target_uid in fiber_elements_by_uid:
                            fiber_uid = target_uid
                            if fiber_uid in connections_map:
                                for final_target_uid in connections_map[fiber_uid]:
                                    if final_target_uid in real_node_uids:
                                        edge_tuple = tuple(sorted((from_uid, final_target_uid)))
                                        if edge_tuple not in processed_edge_tuples:
                                            processed_connections.append({
                                                'from_node': from_uid, 'to_node': final_target_uid,
                                                'fiber_element': fiber_elements_by_uid[fiber_uid]
                                            })
                                            processed_edge_tuples.add(edge_tuple)
                        elif target_uid in real_node_uids:
                            edge_tuple = tuple(sorted((from_uid, target_uid)))
                            if edge_tuple not in processed_edge_tuples:
                                processed_connections.append({
                                    'from_node': from_uid, 'to_node': target_uid, 'fiber_element': None
                                })
                                processed_edge_tuples.add(edge_tuple)

                nodes_to_plot = [el for el in elements if el['uid'] in real_node_uids]
                
                has_geo_data = all(
                    'latitude' in el.get('metadata', {}).get('location', {}) and
                    'longitude' in el.get('metadata', {}).get('location', {}) and
                    not (el['metadata']['location']['latitude'] == 0 and el['metadata']['location']['longitude'] == 0)
                    for el in nodes_to_plot
                ) if nodes_to_plot else False

                if has_geo_data:
                    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(nodes_to_plot, edfa_equipment_data)
                    node_coords = {el['uid']: el['metadata']['location'] for el in nodes_to_plot}

                    # Prepare lists for a single trace of invisible hover points for all fiber spans
                    hover_lats, hover_lons, map_hover_texts = [], [], []

                    for conn in processed_connections:
                        from_node_uid, to_node_uid = conn['from_node'], conn['to_node']
                        if from_node_uid in node_coords and to_node_uid in node_coords:
                            from_coords, to_coords = node_coords[from_node_uid], node_coords[to_node_uid]
                            
                            # Draw the visible line for the connection
                            fig.add_trace(go.Scattermapbox(
                                lat=[from_coords['latitude'], to_coords['latitude']],
                                lon=[from_coords['longitude'], to_coords['longitude']],
                                mode='lines', line=dict(width=2, color='gray'),
                                hoverinfo='none'
                            ))
                            
                            # Collect data for the invisible hover point to be added in a single batch
                            fiber_element = conn.get('fiber_element')
                            if fiber_element:
                                hover_lats.append((from_coords['latitude'] + to_coords['latitude']) / 2)
                                hover_lons.append((from_coords['longitude'] + to_coords['longitude']) / 2)
                                map_hover_texts.append(get_element_tooltip_text(fiber_element, edfa_equipment_data))
                    
                    # Add one trace for all invisible hover points. This is more efficient and reliable.
                    if map_hover_texts:
                        fig.add_trace(go.Scattermapbox(
                            lat=hover_lats, lon=hover_lons, mode='markers',
                            marker=dict(size=20, color='rgba(0,0,0,0)'),
                            hovertext=map_hover_texts,
                            hovertemplate='%{hovertext}<extra></extra>'
                        ))

                    # Add the visible nodes on top
                    fig.add_trace(go.Scattermapbox(
                        lat=[el['metadata']['location']['latitude'] for el in nodes_to_plot],
                        lon=[el['metadata']['location']['longitude'] for el in nodes_to_plot],
                        mode='markers+text',
                        marker=dict(size=15, color=node_colors, symbol=node_symbols),
                        text=[el['uid'] for el in nodes_to_plot],
                        hovertext=node_hover_texts,
                        hovertemplate='%{hovertext}<extra></extra>'
                    ))

                    fig.update_layout(
                        title_text=data.get('network_name', 'Topología de Red (Geolocalizada)'),
                        showlegend=False, mapbox_style="open-street-map",
                        margin={"r": 0, "t": 40, "l": 0, "b": 0},
                        mapbox=dict(
                            center=dict(lat=sum(n['metadata']['location']['latitude'] for n in nodes_to_plot)/len(nodes_to_plot),
                                        lon=sum(n['metadata']['location']['longitude'] for n in nodes_to_plot)/len(nodes_to_plot)),
                            zoom=5
                        ))
                else:
                    node_hover_texts, node_symbols, node_colors = get_node_styles_and_tooltips(nodes_to_plot, edfa_equipment_data)
                    node_coords = {el['uid']: el['metadata']['location'] for el in nodes_to_plot}
                    
                    # Prepare lists for a single trace of invisible hover points for all fiber spans
                    hover_x, hover_y, d2_hover_texts = [], [], []

                    for conn in processed_connections:
                        from_uid, to_uid = conn['from_node'], conn['to_node']
                        if from_uid in node_coords and to_uid in node_coords:
                            from_coords, to_coords = node_coords[from_uid], node_coords[to_uid]
                            
                            # Draw the visible line
                            fig.add_trace(go.Scatter(
                                x=[from_coords.get('longitude'), to_coords.get('longitude')],
                                y=[from_coords.get('latitude'), to_coords.get('latitude')],
                                mode='lines', line=dict(width=2, color='gray'),
                                hoverinfo='none'
                            ))
                            
                            # Collect data for the invisible hover point to be added in a single batch
                            fiber_element = conn.get('fiber_element')
                            if fiber_element:
                                hover_x.append((from_coords.get('longitude') + to_coords.get('longitude')) / 2)
                                hover_y.append((from_coords.get('latitude') + to_coords.get('latitude')) / 2)
                                d2_hover_texts.append(get_element_tooltip_text(fiber_element, edfa_equipment_data))

                    # Add one trace for all invisible hover points. This is more efficient and reliable.
                    if d2_hover_texts:
                        fig.add_trace(go.Scatter(
                            x=hover_x, y=hover_y, mode='markers',
                            marker=dict(size=20, color='rgba(0,0,0,0)'),
                            hovertext=d2_hover_texts,
                            hovertemplate='%{hovertext}<extra></extra>'
                        ))

                    # Add the visible nodes on top
                    node_x = [el['metadata']['location']['longitude'] for el in nodes_to_plot]
                    node_y = [el['metadata']['location']['latitude'] for el in nodes_to_plot]
                    node_text_on_graph = [el['uid'] for el in nodes_to_plot]

                    text_positions = ['bottom center'] * len(nodes_to_plot)
                    try:
                        first_transceiver_index = next(i for i, el in enumerate(nodes_to_plot) if el.get('type') == 'Transceiver')
                        text_positions[first_transceiver_index] = 'top center'
                    except StopIteration:
                        pass

                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y, text=node_text_on_graph, hovertext=node_hover_texts,
                        hovertemplate='%{hovertext}<extra></extra>', mode='markers+text',
                        textposition=text_positions,
                        marker=dict(size=20, color=node_colors, symbol=node_symbols),
                        textfont=dict(size=9)
                    ))
                    
                    fig.update_layout(
                        title_text=data.get('network_name', 'Topología de Red'),
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        hovermode='closest', plot_bgcolor='#f0f0f0',
                        margin=dict(l=0, r=0, t=40, b=0), autosize=True
                    )

                graph_json = fig.to_json()
                return render_template('scenario2.html', graph_json=graph_json)
            except Exception as e:
                return render_template('scenario2.html', error=f"Error al procesar el archivo: {e}")
        else:
            return render_template('scenario2.html', error="Tipo de archivo inválido. Por favor, suba un archivo .json")
            
    return render_template('scenario2.html')

@app.route('/scenario03')
def scenario03():
    """Renders the scenario 3 page."""
    return render_template('scenario3.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.get_json()
    
    # Extract parameters from the JSON payload sent by the frontend
    tx_power_dbm = float(data.get('tx_power_dbm', 30))
    sensitivity_receiver_dbm = float(data.get('sensitivity_receiver_dbm', 14))
    fiber_params_from_frontend = data.get('fiber_params', [])
    params_for_calc = {
        'tx_power_dbm': tx_power_dbm,
        'sensitivity_receiver_dbm': sensitivity_receiver_dbm,
        'fiber_params': fiber_params_from_frontend # Already structured by frontend
    }

    resultados = calcular_red(params_for_calc)
    # `resultados` now includes 'plot_dbm_plotly' and 'plot_linear_plotly'
    
    return jsonify(resultados) # Send all results, including Plotly data, back to frontend

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
