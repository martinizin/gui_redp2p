from flask import Flask, jsonify, render_template, request
from red import calcular_red, obtener_topologia_datos  # Importamos la función de cálculo de la red

app = Flask(__name__)

@app.route('/')
def index():
    # Tomamos la longitud de la fibra desde el formulario
    fibra_length = request.args.get('fibra_length', default=100, type=float)  # Valor por defecto: 100 km
    
    # Llamamos a la función para realizar los cálculos y obtener los resultados
    resultados = calcular_red(fiber_length=fibra_length)
    
    return render_template('index.html', resultados=resultados)

@app.route('/red')
def obtener_topologia():
    # Llamamos a la función para obtener solo la topología (nodos y enlaces)
    nodos, enlaces = obtener_topologia_datos()
    
    # Agregar los parámetros de cada nodo (potencia, etc.)
    for nodo in nodos:
        if nodo['label'] == 'Emisor':
            nodo['potencia'] = 6  # Ejemplo de potencia de emisión en dBm
        elif nodo['label'] == 'Receptor':
            nodo['potencia'] = -10  # Ejemplo de potencia recibida en dBm
        elif nodo['label'] == 'Fibra':
            nodo['potencia'] = -5  # Potencia de la fibra (atenuación, por ejemplo)
        else:
            nodo['potencia'] = 0  # En caso de otros nodos (si es que existen)
    
    return jsonify({"nodos": nodos, "enlaces": enlaces})

if __name__ == '__main__':
    app.run(debug=True)
