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

    # Initial parameters for the first calculation (default view)
    # These should align with what your frontend 'dibujarRed' initializes with
    # Or be the basis for that initialization.
    initial_fiber_params = []
    # Attempt to derive initial fiber params from the default topology if possible
    # This example assumes a simple single fiber span if no complex logic in obtener_topologia_datos
    # You might need to adjust this based on how `obtener_topologia_datos` and your JS `dibujarRed` are set up.
    
    # Extract default fiber parameters from the initial network setup in JS (if possible)
    # For now, let's use a simplified default, similar to your previous 'initial_params'
    default_span_length = 5 # Default length for initial calculation
    # This should ideally match the initial setup in index.html's dibujarRed or be passed to it.
    
    # We need to create a default set of fiber spans if `obtener_topologia_datos`
    # doesn't provide enough detail for `calcular_red`'s `fiber_params`.
    # The goal is to have `initial_params` that `calcular_red` can process.

    # Let's assume `obtener_topologia_datos` gives us some basic structure.
    # We will adapt it or set a fixed default for the initial graph.
    # The 'enlaces' from `obtener_topologia_datos` in `red.py` might not have all 'length_stretch' etc.
    # For the initial graph, we'll use a default similar to your previous setup.

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

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.get_json()
    
    # Extract parameters from the JSON payload sent by the frontend
    tx_power_dbm = float(data.get('tx_power_dbm', 30))
    sensitivity_receiver_dbm = float(data.get('sensitivity_receiver_dbm', 14))
    
    # The fiber_params will come from the Vis.js network data,
    # processed and sent by the frontend.
    fiber_params_from_frontend = data.get('fiber_params', [])
    
    # Ensure fiber_params are in the format expected by calcular_red
    # The frontend should send `length_stretch` and other necessary params.
    # Example of what `red.py` expects for each tramo:
    # {
    # 'length': tramo['length_stretch'],
    # 'loss_coef': tramo['loss_coef'],
    # 'att_in': tramo['att_in'],
    # 'con_in': tramo['con_in'],
    # 'con_out': tramo['con_out']
    # }
    
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
