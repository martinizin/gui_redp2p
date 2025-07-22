from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json

from scenario01 import calcular_red, obtener_topologia_datos
from scenario02 import calculate_scenario02, update_scenario02_parameters, process_scenario02_data, create_topology_visualization_from_data
from scenario03 import handle_scenario03, get_topology_names, get_topology_data, upload_topology_file, update_network_parameters, calculate_routes

# Carga las variables de entorno - specify explicit path for Docker compatibility
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Maneja la página principal con la lógica del escenario 1."""
    # Proporciona datos iniciales para la red y los gráficos
    nodos, enlaces = obtener_topologia_datos()

    initial_params_for_calc = {
        'tx_power_dbm': 30,  # Valor por defecto del JS
        'sensitivity_receiver_dbm': 14,  # Valor por defecto del JS
        'fiber_params': [{  # Un tramo por defecto
            'loss_coef': 0.2,
            'att_in': 0,
            'con_in': 0.25,
            'con_out': 0.30,
            'length_stretch': 23  # Longitud por defecto del formulario JS
        }]
    }
    resultados = calcular_red(initial_params_for_calc)  # Ahora incluye diccionarios de Plotly

    return render_template('index.html',
                           resultados=resultados,  # Contiene datos de gráficos Plotly
                           nodos=nodos,
                           enlaces=enlaces,
                           initial_graph_dbm=resultados.get('plot_dbm_plotly'),
                           initial_graph_linear=resultados.get('plot_linear_plotly'))

@app.route('/scenario02', methods=['GET', 'POST'])
def scenario02():
    """Renderiza la página del escenario 2 y maneja la subida de archivos para visualización de red."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('scenario2.html', error="No se encontró el archivo")
        file = request.files['file']
        result = process_scenario02_data(file) 
        if 'error' in result:
            return render_template('scenario2.html', error=result['error'])
        else:
            return render_template('scenario2.html', 
                                 graph_json=result['graph_json'],
                                 enhanced_data=result.get('enhanced_data'),
                                 is_example_topology=False)
    
    # GET request - Cargar topología de ejemplo automáticamente
    try:
        example_topology_path = 'data/enlace_WDM.json'
        if os.path.exists(example_topology_path):
            # Cargar los datos de la topología de ejemplo
            with open(example_topology_path, 'r', encoding='utf-8') as f:
                topology_data = json.load(f)
            
            # Usar la función de visualización de topología existente
            result = create_topology_visualization_from_data(topology_data)
            
            if 'error' in result:
                return render_template('scenario2.html', error=f"Error al cargar topología de ejemplo: {result['error']}")
            
            # Procesar elementos para obtener enhanced_data
            from scenario02 import enhance_elements_with_parameters
            enhanced_elements = enhance_elements_with_parameters(topology_data.get('elements', []))
            enhanced_data = topology_data.copy()
            enhanced_data['elements'] = enhanced_elements
            
            return render_template('scenario2.html', 
                                 graph_json=result['figure'].to_json(),
                                 enhanced_data=enhanced_data,
                                 is_example_topology=True)
        else:
            # Si no existe el archivo de ejemplo, mostrar la página vacía
            return render_template('scenario2.html', 
                                 error="Archivo de topología de ejemplo no encontrado")
    except Exception as e:
        return render_template('scenario2.html', 
                             error=f"Error al cargar topología de ejemplo: {e}")

@app.route('/scenario03')
def scenario03():
    """Renderiza la página del escenario 3."""
    return handle_scenario03()

# Rutas de API del Escenario 03
@app.route("/get_topology_names")
def get_topology_names_route():
    """Endpoint de API para obtener los archivos de topología disponibles."""
    return get_topology_names()

@app.route("/get_topology")
def get_topology_route():
    """Endpoint de API para obtener datos de topología."""
    return get_topology_data()

@app.route('/upload_topology', methods=['POST'])
def upload_topology_route():
    """Endpoint de API para subir archivos de topología."""
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo en la solicitud'}), 400
    
    file = request.files['file']
    return upload_topology_file(file)

@app.route('/update_network_parameters', methods=['POST'])
def update_network_parameters_route():
    """Endpoint de API para actualizar parámetros de red."""
    return update_network_parameters()

@app.route('/calculate_routes', methods=['POST'])
def calculate_routes_route():
    """Endpoint de API para calcular rutas óptimas entre nodos."""
    return calculate_routes()

@app.route('/update_scenario02_parameters', methods=['POST'])
def update_scenario02_parameters_route():
    """Endpoint de API para actualizar parámetros de red del escenario02."""
    return update_scenario02_parameters()

@app.route('/calculate_scenario02', methods=['POST'])
def calculate_scenario02_route():
    """Endpoint de API para calcular la red del escenario02."""
    return calculate_scenario02()

@app.route('/test_gnpy', methods=['GET'])
def test_gnpy_route():
    """Endpoint de API para probar la integración de gnpy."""
    from scenario02 import test_gnpy_integration
    return jsonify(test_gnpy_integration())

@app.route('/calcular', methods=['POST'])
def calcular():
    """Maneja las solicitudes de cálculo para el escenario 1."""
    data = request.get_json()
    # Extrae parámetros del payload JSON enviado por el frontend
    tx_power_dbm = float(data.get('tx_power_dbm', 30))
    sensitivity_receiver_dbm = float(data.get('sensitivity_receiver_dbm', 14))
    fiber_params_from_frontend = data.get('fiber_params', [])
    params_for_calc = {
        'tx_power_dbm': tx_power_dbm,
        'sensitivity_receiver_dbm': sensitivity_receiver_dbm,
        'fiber_params': fiber_params_from_frontend  # Ya estructurado por el frontend
    }

    resultados = calcular_red(params_for_calc)
    # `resultados` ahora incluye 'plot_dbm_plotly' y 'plot_linear_plotly'
    return jsonify(resultados)  # Envía todos los resultados, incluyendo datos de Plotly, de vuelta al frontend

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
