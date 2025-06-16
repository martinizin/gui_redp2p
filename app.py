from flask import Flask, render_template, request, jsonify
from red import calcular_red, obtener_topologia_datos
import plotly.graph_objects as go

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

@app.route('/scenario02')
def scenario02():
    """Renders the scenario 2 page."""
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
