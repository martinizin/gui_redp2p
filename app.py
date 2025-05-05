from flask import Flask, jsonify, render_template, request
from red import calcular_red, obtener_topologia_datos  # Importamos la función de cálculo de la red

app = Flask(__name__)

@app.route('/')
def index():
    # Tomamos la longitud de la fibra desde el formulario
    fibra_length = request.args.get('fibra_length', default=0, type=float)  # Valor por defecto: 100 km
    
    # Llamamos a la función para realizar los cálculos y obtener los resultados
    resultados = calcular_red(fiber_length=fibra_length)
    
    return render_template('index.html', resultados=resultados)

@app.route('/red')
def obtener_topologia():
    # Llamamos a la función para obtener solo la topología (nodos y enlaces)
    nodos, enlaces = obtener_topologia_datos()
    return jsonify({"nodos": nodos, "enlaces": enlaces})

if __name__ == '__main__':
    app.run(debug=True)
