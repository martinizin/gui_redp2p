from flask import Flask, render_template, request, jsonify
from red import calcular_red, obtener_topologia_datos
import plotly.graph_objects as go
import json

app = Flask(__name__)

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

                # Get all unique node UIDs from connections
                node_uids = {c['from_node'] for c in connections} | {c['to_node'] for c in connections}
                elements_by_uid = {el['uid']: el for el in elements}
                
                # Check if every node defined in connections has valid geolocation data
                has_geo_data = True
                nodes_for_map = []
                for uid in node_uids:
                    node = elements_by_uid.get(uid)
                    if not node:
                        has_geo_data = False
                        break

                    location = node.get('metadata', {}).get('location', {})
                    if not ('latitude' in location and 'longitude' in location) or \
                       (location.get('latitude') == 0 and location.get('longitude') == 0):
                        has_geo_data = False
                        break
                    
                    nodes_for_map.append(node)

                if len(nodes_for_map) != len(node_uids):
                    has_geo_data = False

                if has_geo_data:
                    # Geolocation rendering on a map
                    node_coords = {el['uid']: el['metadata']['location'] for el in nodes_for_map}

                    # Add edges
                    for conn in connections:
                        from_node_uid = conn.get('from_node')
                        to_node_uid = conn.get('to_node')
                        if from_node_uid in node_coords and to_node_uid in node_coords:
                            from_coords = node_coords[from_node_uid]
                            to_coords = node_coords[to_node_uid]
                            fig.add_trace(go.Scattermapbox(
                                lat=[from_coords['latitude'], to_coords['latitude']],
                                lon=[from_coords['longitude'], to_coords['longitude']],
                                mode='lines',
                                line=dict(width=2, color='blue'),
                                hoverinfo='none'
                            ))

                    # Add nodes
                    fig.add_trace(go.Scattermapbox(
                        lat=[el['metadata']['location']['latitude'] for el in nodes_for_map],
                        lon=[el['metadata']['location']['longitude'] for el in nodes_for_map],
                        mode='markers+text',
                        marker=dict(size=15, color='red'),
                        text=[el['uid'] for el in nodes_for_map],
                        hovertext=[f"ID: {el['uid']}<br>Type: {el.get('type', 'N/A')}" for el in nodes_for_map],
                        hovertemplate='%{hovertext}<extra></extra>'
                    ))

                    fig.update_layout(
                        title_text=data.get('network_name', 'Topología de Red (Geolocalizada)'),
                        showlegend=False,
                        mapbox_style="open-street-map",
                        margin={"r": 0, "t": 40, "l": 0, "b": 0},
                        mapbox=dict(
                            center=dict(
                                lat=sum(el['metadata']['location']['latitude'] for el in nodes_for_map) / len(nodes_for_map),
                                lon=sum(el['metadata']['location']['longitude'] for el in nodes_for_map) / len(nodes_for_map)
                            ),
                            zoom=5
                        )
                    )
                else:
                    # Fallback to 2D plot
                    nodes_with_meta = [el for el in elements if 'location' in el.get('metadata', {})]
                    node_coords = {el['uid']: el['metadata']['location'] for el in nodes_with_meta}
                    # Existing 2D plotting logic
                    for conn in connections:
                        from_uid, to_uid = conn.get('from_node'), conn.get('to_node')
                        if from_uid in node_coords and to_uid in node_coords:
                            from_coords = node_coords[from_uid]
                            to_coords = node_coords[to_uid]
                            fig.add_trace(go.Scatter(
                                x=[from_coords.get('longitude'), to_coords.get('longitude')],
                                y=[from_coords.get('latitude'), to_coords.get('latitude')],
                                mode='lines',
                                line=dict(width=1, color='gray'),
                                hoverinfo='none'
                            ))
                    
                    node_x = [el['metadata']['location']['longitude'] for el in nodes_with_meta]
                    node_y = [el['metadata']['location']['latitude'] for el in nodes_with_meta]
                    node_text_on_graph = [el['uid'] for el in nodes_with_meta]
                    node_hover_text = [f"ID: {el['uid']}<br>Type: {el.get('type', 'N/A')}" for el in nodes_with_meta]

                    text_positions = ['bottom center'] * len(nodes_with_meta)
                    try:
                        first_transceiver_index = next(i for i, el in enumerate(nodes_with_meta) if el.get('type') == 'Transceiver')
                        text_positions[first_transceiver_index] = 'top center'
                    except StopIteration:
                        pass

                    type_colors = {'Transceiver': '#ff6347', 'Fiber': '#add8e6', 'Edfa': '#90ee90', 'default': '#808080'}
                    node_colors = [type_colors.get(el.get('type'), 'default') for el in nodes_with_meta]

                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        text=node_text_on_graph,
                        hovertext=node_hover_text,
                        hovertemplate='%{hovertext}<extra></extra>',
                        mode='markers+text',
                        textposition=text_positions,
                        marker=dict(size=20, color=node_colors, symbol='circle'),
                        textfont=dict(size=9)
                    ))
                    
                    fig.update_layout(
                        title_text=data.get('network_name', 'Topología de Red'),
                        showlegend=False,
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        hovermode='closest',
                        plot_bgcolor='#f0f0f0',
                        margin=dict(l=0, r=0, t=40, b=0),
                        autosize=True
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
    app.run(debug=True)
