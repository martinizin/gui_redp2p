import numpy as np
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, Transceiver
from gnpy.core.utils import watt2dbm, db2lin
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def dbm2mw(dbm):
    return 10 ** (dbm / 10)

# --- Funciones auxiliares para inputs robustos (opcional para CLI) ---

def solicitar_float(mensaje, valor_por_defecto=None, minimo=None, maximo=None):
    while True:
        try:
            entrada = input(f"{mensaje} " + (f"[Por defecto: {valor_por_defecto}]: " if valor_por_defecto is not None else ": "))
            if entrada == "" and valor_por_defecto is not None:
                return valor_por_defecto
            valor = float(entrada)
            if minimo is not None and valor < minimo:
                print(f"El valor debe ser al menos {minimo}. Intenta de nuevo.")
                continue
            if maximo is not None and valor > maximo:
                print(f"El valor debe ser máximo {maximo}. Intenta de nuevo.")
                continue
            return valor
        except ValueError:
            print("Entrada inválida. Ingresa un número válido.")

def solicitar_int(mensaje, minimo=None, maximo=None):
    while True:
        try:
            valor = int(input(mensaje + ": "))
            if minimo is not None and valor < minimo:
                print(f"El valor debe ser al menos {minimo}. Intenta de nuevo.")
                continue
            if maximo is not None and valor > maximo:
                print(f"El valor debe ser máximo {maximo}. Intenta de nuevo.")
                continue
            return valor
        except ValueError:
            print("Entrada inválida. Ingresa un número entero válido.")

def pedir_parametros_tramo(tramo_num, valores_default):
    print(f"\nConfiguración del tramo {tramo_num}:")
    cambiar = input("¿Desea cambiar los parámetros de la fibra de este tramo? (s/n): ").lower()
    if cambiar == "s":
        loss_coef = solicitar_float("Coeficiente de pérdida (dB/km)", valores_default['loss_coef'], minimo=0)
        att_in = solicitar_float("Pérdida atenuación interna (dB)", valores_default['att_in'], minimo=0)
        con_in = solicitar_float("Pérdida conector entrada (dB)", valores_default['con_in'], minimo=0)
        con_out = solicitar_float("Pérdida conector salida (dB)", valores_default['con_out'], minimo=0)
        return {
            'loss_coef': loss_coef,
            'att_in': att_in,
            'con_in': con_in,
            'con_out': con_out
        }
    else:
        return valores_default.copy()

# --- Función de simulación detallada ---

def simular_red_por_tramos_detallado(tx_power_dbm=16.53,
                                     sensibilidad_receptor_dbm=-28,
                                     tramos_params=None):
    if tramos_params is None:
        raise ValueError("Debes proporcionar los parámetros de los tramos")

    power_history_dbm = [tx_power_dbm]
    power_history_linear = [db2lin(tx_power_dbm)]
    longitud_acumulada = [0]
    current_power_dbm = tx_power_dbm

    num_tramos = len(tramos_params)
    detalles_tramos = []

    for i, params in enumerate(tramos_params):
        longitud_tramo = params.get('length', 5)
        loss_coef = params.get('loss_coef', 0.2)
        att_in = params.get('att_in', 0)
        con_in = params.get('con_in', 0.25)
        con_out = params.get('con_out', 0.30)

        num_segmentos = int(np.ceil(longitud_tramo / 5))
        potencia_inicial_tramo = current_power_dbm

        for j in range(num_segmentos):
            segment_length = min(5, longitud_tramo - j * 5)
            attenuation_segment_db = loss_coef * segment_length
            current_power_dbm -= attenuation_segment_db
            power_history_dbm.append(current_power_dbm)
            power_history_linear.append(db2lin(current_power_dbm))
            longitud_acumulada.append(longitud_acumulada[-1] + segment_length)

        # Conectores y atenuaciones al final del tramo
        current_power_dbm -= con_out
        power_history_dbm.append(current_power_dbm)
        power_history_linear.append(db2lin(current_power_dbm))
        longitud_acumulada.append(longitud_acumulada[-1])

        current_power_dbm -= con_in
        power_history_dbm.append(current_power_dbm)
        power_history_linear.append(db2lin(current_power_dbm))
        longitud_acumulada.append(longitud_acumulada[-1])

        current_power_dbm -= att_in
        power_history_dbm.append(current_power_dbm)
        power_history_linear.append(db2lin(current_power_dbm))
        longitud_acumulada.append(longitud_acumulada[-1])

        potencia_final_tramo = power_history_dbm[-1]
        atenuacion_tramo = potencia_inicial_tramo - potencia_final_tramo
        longitud_acum_tramo = longitud_acumulada[-1]

        detalles_tramos.append({
            'numero': i + 1,
            'longitud_acumulada': longitud_acum_tramo,
            'potencia_final_dbm': potencia_final_tramo,
            'atenuacion_tramo': atenuacion_tramo,
            'longitud_tramo': longitud_tramo,
        })

    resultados = {
        'power_history_dbm': power_history_dbm,
        'power_history_linear': power_history_linear,
        'longitud_acumulada': longitud_acumulada,
        'potencia_final_dbm': power_history_dbm[-1],
        'atenuacion_total': tx_power_dbm - power_history_dbm[-1],
        'sensibilidad_receptor_dbm': sensibilidad_receptor_dbm,
        'detalles_tramos': detalles_tramos,
        'potencia_inicial_dbm': tx_power_dbm,
        'longitud_total': sum([d['longitud_tramo'] for d in detalles_tramos])
    }

    return resultados

# --- Función para graficar resultados ---

def graficar_potencia(longitud_acumulada, power_history_dbm, power_history_linear, sensibilidad_receptor_dbm):
    x_dbm_unique = sorted(list(set([round(l, 2) for l in longitud_acumulada])))
    y_dbm_unique = []
    for length in x_dbm_unique:
        indices = [i for i, l in enumerate([round(val, 2) for val in longitud_acumulada]) if l == length]
        y_dbm_unique.append(power_history_dbm[indices[-1]])

    plt.figure(figsize=(12, 6))
    plt.plot(x_dbm_unique, y_dbm_unique, marker='o', linestyle='-', label="Potencia (dBm)")
    plt.xlabel("Longitud acumulada de la fibra (km)")
    plt.ylabel("Potencia de la señal (dBm)")
    plt.title("Potencia de salida (dBm) vs. Longitud")
    plt.grid(True)
    plt.xticks(np.arange(0, max(x_dbm_unique) + 5, 5))
    plt.axhline(sensibilidad_receptor_dbm, color='r', linestyle='--', label=f'Sensibilidad del receptor: {sensibilidad_receptor_dbm:.2f} dBm')
    plt.legend()
    plt.show()

    x_linear_unique = sorted(list(set([round(l, 2) for l in longitud_acumulada])))
    y_linear_unique = []
    for length in x_linear_unique:
        indices = [i for i, l in enumerate([round(val, 2) for val in longitud_acumulada]) if l == length]
        y_linear_unique.append(power_history_linear[indices[-1]])

    plt.figure(figsize=(12, 6))
    plt.plot(x_linear_unique, y_linear_unique, marker='o', linestyle='-', label="Potencia (lineal)")
    plt.xlabel("Longitud acumulada de la fibra (km)")
    plt.ylabel("Potencia de la señal (Watts)")
    plt.title("Potencia de salida (lineal) vs. Longitud")
    plt.grid(True)
    plt.xticks(np.arange(0, max(x_linear_unique) + 5, 5))
    sensibilidad_lineal_watts = db2lin(sensibilidad_receptor_dbm)
    plt.axhline(sensibilidad_lineal_watts, color='r', linestyle='--', label=f'Sensibilidad del receptor: {sensibilidad_lineal_watts:.2e} Watts')
    plt.ylim(min(y_linear_unique) * 0.9, max(y_linear_unique) * 1.1)
    plt.legend()
    plt.show()

# --- Nueva función para graficar con Plotly (Lineal) ---
def graficar_potencia_plotly_linear(longitud_acumulada, power_history_linear, sensibilidad_receptor_dbm, tx_power_dbm):
    """
    Genera los datos para un gráfico Plotly de potencia lineal vs. longitud acumulada.
    Devuelve una figura de Plotly como un diccionario.
    """
    sensibilidad_receptor_watts = db2lin(sensibilidad_receptor_dbm)
    initial_power_watts = db2lin(tx_power_dbm)

    # Convert power history from Watts to mW for this graph
    power_history_mw = [p / 100 for p in power_history_linear]
    sensibilidad_receptor_mw = sensibilidad_receptor_watts / 100

    # Asegurar que todos los historiales tengan la misma longitud que longitud_acumulada
    # Esto es importante si los eventos de pérdida de conector no incrementan la longitud
    # y queremos que cada punto de datos tenga una coordenada x correspondiente.
    
    # Crear listas únicas para x y para y, tomando el último valor de potencia para cada longitud única.
    # Esto maneja los puntos verticales en las gráficas debido a pérdidas instantáneas (conectores).
    unique_lengths_map = {}
    for i, l in enumerate(longitud_acumulada):
        unique_lengths_map[round(l, 5)] = power_history_mw[i] # Use mW history
    
    sorted_unique_lengths = sorted(unique_lengths_map.keys())
    unique_power_history_mw = [unique_lengths_map[l] for l in sorted_unique_lengths] # Use mW history


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_unique_lengths,
        y=unique_power_history_mw, # Use mW data for y-axis
        mode='lines+markers',
        name='Potencia (mW)' # Update label
    ))

    fig.add_hline(
        y=sensibilidad_receptor_mw, # Use mW for sensitivity line
        line_dash="dash",
        annotation_text=f"Sensibilidad Receptor: {sensibilidad_receptor_mw:.2e} mW", # Update annotation
        annotation_position="bottom right",
        line_color='red'
    )

    fig.update_layout(
        title="Potencia de Salida (Lineal) vs. Longitud",
        xaxis_title="Longitud acumulada de la fibra (km)",
        yaxis_title="Potencia de la señal (mW)", # Update y-axis title
        yaxis_type="linear", 
        legend_title_text='Leyenda',
        height=500
    )
    # Plotly's autorange should handle the new mW scale well.
    # If specific range adjustments are needed later, they can be added to yaxis layout:
    # yaxis_range=[min_val_mw, max_val_mw] 
    return fig.to_dict()

# --- Nueva función para graficar con Plotly (dBm) ---
def graficar_potencia_plotly_dbm(longitud_acumulada, power_history_dbm, sensibilidad_receptor_dbm, tx_power_dbm):
    """
    Genera los datos para un gráfico Plotly de potencia dBm vs. longitud acumulada.
    Devuelve una figura de Plotly como un diccionario.
    """
    # Crear listas únicas para x y para y, tomando el último valor de potencia para cada longitud única.
    unique_lengths_map = {}
    for i, l in enumerate(longitud_acumulada):
        unique_lengths_map[round(l, 5)] = power_history_dbm[i]

    sorted_unique_lengths = sorted(unique_lengths_map.keys())
    unique_power_history_dbm = [unique_lengths_map[l] for l in sorted_unique_lengths]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_unique_lengths,
        y=unique_power_history_dbm,
        mode='lines+markers',
        name='Potencia (dBm)'
    ))

    fig.add_hline(
        y=sensibilidad_receptor_dbm,
        line_dash="dash",
        annotation_text=f"Sensibilidad Receptor: {sensibilidad_receptor_dbm:.2f} dBm",
        annotation_position="bottom right",
        line_color='red'
    )

    fig.update_layout(
        title="Potencia de Salida (dBm) vs. Longitud",
        xaxis_title="Longitud acumulada de la fibra (km)",
        yaxis_title="Potencia de la señal (dBm)",
        legend_title_text='Leyenda',
        height=500 # Altura del gráfico
    )
    return fig.to_dict()

# --- Función para obtener la topología de la red (puedes adaptar según tu app) ---

def obtener_topologia_datos():
    nodos = [
        {"id": 1, "label": "Emisor", "x": 100, "y": 100, "potencia": 6},
        {"id": 2, "label": "Fibra", "x": 300, "y": 100, "potencia": -5, "loss_coef": 0.2, "att_in": 0, "con_in": 0.25, "con_out": 0.30, "longitud": 5},
        {"id": 3, "label": "Receptor", "x": 500, "y": 100, "potencia": -10, "sensibilidad_receptor_dbm": -28}
    ]
    enlaces = [
        {"from": 1, "to": 2, "label": "Fibra 1", "longitud": 5},
        {"from": 2, "to": 3, "label": "Fibra 2", "longitud": 5}
    ]
    return nodos, enlaces

# --- Función principal para la web: calcular_red ---

def calcular_red(params):
    """
    Calcula la propagación de la señal en la red óptica basado en los parámetros proporcionados.
    Espera un diccionario con:
        - tx_power_dbm: Potencia del transmisor en dBm
        - sensitivity_receiver_dbm: Sensibilidad del receptor en dBm
        - fiber_params: Lista de diccionarios para cada tramo de fibra, cada uno con:
            - loss_coef: Coeficiente de pérdida en dB/km
            - att_in: Atenuación en la entrada (dB)
            - con_in: Pérdida del conector de entrada (dB)
            - con_out: Pérdida del conector de salida (dB)
            - length_stretch: Longitud del tramo (km)
    """
    # Adaptar los nombres de los parámetros para la función detallada
    tramos_params = []
    for tramo in params['fiber_params']:
        tramos_params.append({
            'length': tramo['length_stretch'],
            'loss_coef': tramo['loss_coef'],
            'att_in': tramo['att_in'],
            'con_in': tramo['con_in'],
            'con_out': tramo['con_out']
        })

    resultados = simular_red_por_tramos_detallado(
        tx_power_dbm=params['tx_power_dbm'],
        sensibilidad_receptor_dbm=params['sensitivity_receiver_dbm'],
        tramos_params=tramos_params
    )

    # Generar datos para gráficos Plotly
    resultados['plot_linear_plotly'] = graficar_potencia_plotly_linear(
        resultados['longitud_acumulada'],
        resultados['power_history_linear'],
        resultados['sensibilidad_receptor_dbm'],
        resultados['potencia_inicial_dbm']
    )
    resultados['plot_dbm_plotly'] = graficar_potencia_plotly_dbm(
        resultados['longitud_acumulada'],
        resultados['power_history_dbm'],
        resultados['sensibilidad_receptor_dbm'],
        resultados['potencia_inicial_dbm']
    )

    return resultados