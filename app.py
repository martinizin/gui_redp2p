from flask import Flask, render_template, request, jsonify
from red import calcular_red, obtener_topologia_datos
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import io
import base64

app = Flask(__name__)

def dbm2mw(dbm):
    """Convierte dBm a mW."""
    return 10 ** (dbm / 10)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = None
    nodos, enlaces = obtener_topologia_datos()

    if request.method == 'POST':
        try:
            # Obtener parámetros del formulario
            params = {
                'tx_power_dbm': float(request.form['tx_power_dbm']),
                'tx_power_watts': float(request.form['tx_power_watts']),
                'sensitivity_receiver_dbm': float(request.form['sensitivity_receiver_dbm']),
                'fiber_params': []
            }

            # Obtener parámetros de cada fibra
            num_fibers = int(request.form['num_fibers'])
            for i in range(num_fibers):
                fiber_id = f'fiber_{i+1}_'
                params['fiber_params'].append({
                    'loss_coef': float(request.form[fiber_id + 'loss_coef']),
                    'att_in': float(request.form[fiber_id + 'att_in']),
                    'con_in': float(request.form[fiber_id + 'con_in']),
                    'con_out': float(request.form[fiber_id + 'con_out']),
                    'length_stretch': float(request.form[fiber_id + 'length_stretch'])
                })

            # Realizar los cálculos de la red
            resultados = calcular_red(params)

            # Verificar los resultados en el backend
            print("RESULTADOS:", resultados)

            # Generar gráficas (potencia vs longitud)
            fig, ax = plt.subplots()
            ax.plot(resultados['longitud_acumulada'], resultados['power_history_dbm'], label="Potencia (dBm)")
            ax.set_xlabel('Longitud (km)')
            ax.set_ylabel('Potencia (dBm)')
            ax.set_title('Potencia vs Longitud')
            ax.legend()

            # Convertir la gráfica a base64 para mostrarla en el HTML
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_dbm = base64.b64encode(img.getvalue()).decode('utf-8')

            # Gráfica de potencia lineal
            fig, ax = plt.subplots()
            ax.plot(resultados['longitud_acumulada'], resultados['power_history_linear'], label="Potencia (mW)")
            ax.set_xlabel('Longitud (km)')
            ax.set_ylabel('Potencia (mW)')
            ax.set_title('Potencia vs Longitud')
            ax.legend()

            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_linear = base64.b64encode(img.getvalue()).decode('utf-8')

            # Incluir las gráficas en los resultados
            resultados['plot_dbm'] = plot_dbm
            resultados['plot_linear'] = plot_linear

        except Exception as e:
            print(f"Error al procesar el formulario: {e}")

    # Generar gráficos iniciales si no hay resultados
    if not resultados:
        # Valores iniciales
        initial_params = {
            'tx_power_dbm': 30,
            'tx_power_watts': dbm2mw(30),
            'sensitivity_receiver_dbm': -28,
            'fiber_params': [{
                'loss_coef': 0.2,
                'att_in': 0,
                'con_in': 0.25,
                'con_out': 0.30,
                'length_stretch': 5
            }]
        }
        resultados = calcular_red(initial_params)
        
        # Generar gráficas iniciales
        fig, ax = plt.subplots()
        ax.plot(resultados['longitud_acumulada'], resultados['power_history_dbm'], label="Potencia (dBm)")
        ax.set_xlabel('Longitud (km)')
        ax.set_ylabel('Potencia (dBm)')
        ax.set_title('Potencia vs Longitud')
        ax.legend()

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        resultados['plot_dbm'] = base64.b64encode(img.getvalue()).decode('utf-8')

        fig, ax = plt.subplots()
        ax.plot(resultados['longitud_acumulada'], resultados['power_history_linear'], label="Potencia (mW)")
        ax.set_xlabel('Longitud (km)')
        ax.set_ylabel('Potencia (mW)')
        ax.set_title('Potencia vs Longitud')
        ax.legend()

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        resultados['plot_linear'] = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('index.html', resultados=resultados, nodos=nodos, enlaces=enlaces)

@app.route('/red', methods=['GET'])
def obtener_topologia():
    # Obtener la topología de la red (nodos y enlaces)
    nodos, enlaces = obtener_topologia_datos()
    return jsonify({"nodos": nodos, "enlaces": enlaces})

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.get_json()
    # Parse parameters from data as needed
    # Example:
    tx_power_dbm = float(data.get('tx_power_dbm', 30))
    tx_power_watts_input = float(data.get('tx_power_watts_input', 1))
    sensitivity_receiver_dbm = float(data.get('sensitivity_receiver_dbm', 14))
    fiber_params = data.get('fiber_params', [])

    # Call your calculation function
    resultados = calcular_red({
        'tx_power_dbm': tx_power_dbm,
        'tx_power_watts_input': tx_power_watts_input,
        'sensitivity_receiver_dbm': sensitivity_receiver_dbm,
        'fiber_params': fiber_params
    })

    return jsonify(resultados)
if __name__ == '__main__':
    app.run(debug=True)
