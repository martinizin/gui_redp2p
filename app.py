from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from scenario01 import calcular_red, obtener_topologia_datos
from scenario02 import process_scenario02_data
from scenario03 import handle_scenario03, get_topology_names, get_topology_data, upload_topology_file

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Handles the main page with scenario 1 logic."""
    # Provide initial data for the network and graphs
    nodos, enlaces = obtener_topologia_datos()

    initial_params_for_calc = {
        'tx_power_dbm': 30,  # Default from your JS
        'sensitivity_receiver_dbm': 14,  # Default from your JS
        'fiber_params': [{  # A default single span
            'loss_coef': 0.2,
            'att_in': 0,
            'con_in': 0.25,
            'con_out': 0.30,
            'length_stretch': 23  # Default length from your JS form
        }]
    }
    resultados = calcular_red(initial_params_for_calc)  # This now includes Plotly dicts

    return render_template('index.html',
                           resultados=resultados,  # Contains Plotly graph data
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
        result = process_scenario02_data(file)
        
        if 'error' in result:
            return render_template('scenario2.html', error=result['error'])
        else:
            return render_template('scenario2.html', graph_json=result['graph_json'])

    return render_template('scenario2.html')

@app.route('/scenario03')
def scenario03():
    """Renders the scenario 3 page."""
    return handle_scenario03()

# Scenario 03 API routes
@app.route("/get_topology_names")
def get_topology_names_route():
    """API endpoint to get available topology files."""
    return get_topology_names()

@app.route("/get_topology")
def get_topology_route():
    """API endpoint to get topology data."""
    return get_topology_data()

@app.route('/upload_topology', methods=['POST'])
def upload_topology_route():
    """API endpoint to upload topology files."""
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo en la solicitud'}), 400
    
    file = request.files['file']
    return upload_topology_file(file)

@app.route('/calcular', methods=['POST'])
def calcular():
    """Handles the calculation requests for scenario 1."""
    data = request.get_json()
    # Extract parameters from the JSON payload sent by the frontend
    tx_power_dbm = float(data.get('tx_power_dbm', 30))
    sensitivity_receiver_dbm = float(data.get('sensitivity_receiver_dbm', 14))
    fiber_params_from_frontend = data.get('fiber_params', [])
    params_for_calc = {
        'tx_power_dbm': tx_power_dbm,
        'sensitivity_receiver_dbm': sensitivity_receiver_dbm,
        'fiber_params': fiber_params_from_frontend  # Already structured by frontend
    }

    resultados = calcular_red(params_for_calc)
    # `resultados` now includes 'plot_dbm_plotly' and 'plot_linear_plotly'
    
    return jsonify(resultados)  # Send all results, including Plotly data, back to frontend

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
