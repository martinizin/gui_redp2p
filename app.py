from flask import Flask, render_template, request, jsonify
from red import calcular_red, obtener_topologia_datos
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = None
    nodos, enlaces = obtener_topologia_datos()

    if request.method == 'POST':
        try:
            # Captura de los datos del formulario
            unidad_potencia = request.form['unit']
            tx_power = float(request.form['tx_power'])
            receiver_sensitivity = float(request.form['receiver_sensitivity'])
            num_segments = int(request.form['num_segments'])
            segment_lengths = []

            for i in range(num_segments):
                segment_length = float(request.form.get(f'segment_{i+1}', 0))
                segment_lengths.append(segment_length)

            # Realizar los cálculos de la red
            resultados = calcular_red(tx_power, receiver_sensitivity, num_segments, segment_lengths)

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

    return render_template('index.html', resultados=resultados, nodos=nodos, enlaces=enlaces)

@app.route('/red', methods=['GET'])
def obtener_topologia():
    # Obtener la topología de la red (nodos y enlaces)
    nodos, enlaces = obtener_topologia_datos()
    return jsonify({"nodos": nodos, "enlaces": enlaces})

if __name__ == '__main__':
    app.run(debug=True)
