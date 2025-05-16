import numpy as np
import matplotlib.pyplot as plt
from gnpy.core.utils import watt2dbm, db2lin
import io
import base64
import plotly.graph_objs as go
import json

def simular_red_por_tramos_detallado(tx_power_dbm=16.53,
                                     sensibilidad_receptor_dbm=-28,
                                     tramos_params=None):
    if tramos_params is None:
        raise ValueError("Debes proporcionar los parámetros de los tramos")

    power_history_dbm = [tx_power_dbm]
    power_history_linear = [db2lin(tx_power_dbm)]
    longitud_acumulada = [0]
    current_power_dbm = tx_power_dbm

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

        if i < len(tramos_params) - 1:
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
        else:
            # Último tramo también aplica pérdidas
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
    }

    return resultados

def graficar_potencia_plotly(resultados):
    # Gráfica potencia (dBm) vs longitud
    trace_dbm = go.Scatter(
        x=resultados['longitud_acumulada'],
        y=resultados['power_history_dbm'],
        mode='lines+markers',
        name='Potencia (dBm)'
    )
    layout_dbm = go.Layout(
        title='Potencia vs Longitud (dBm)',
        xaxis=dict(title='Longitud acumulada (km)'),
        yaxis=dict(title='Potencia (dBm)'),
        shapes=[  # Línea de sensibilidad receptor
            dict(
                type='line',
                xref='paper',
                x0=0,
                x1=1,
                y0=resultados['sensibilidad_receptor_dbm'],
                y1=resultados['sensibilidad_receptor_dbm'],
                line=dict(color='red', dash='dash')
            )
        ]
    )
    fig_dbm = go.Figure(data=[trace_dbm], layout=layout_dbm)

    # Gráfica potencia lineal (mW) vs longitud
    trace_mw = go.Scatter(
        x=resultados['longitud_acumulada'],
        y=resultados['power_history_linear'],
        mode='lines+markers',
        name='Potencia (mW)'
    )
    layout_mw = go.Layout(
        title='Potencia vs Longitud (mW)',
        xaxis=dict(title='Longitud acumulada (km)'),
        yaxis=dict(title='Potencia (mW)'),
        shapes=[
            dict(
                type='line',
                xref='paper',
                x0=0,
                x1=1,
                y0=db2lin(resultados['sensibilidad_receptor_dbm']),
                y1=db2lin(resultados['sensibilidad_receptor_dbm']),
                line=dict(color='red', dash='dash')
            )
        ]
    )
    fig_mw = go.Figure(data=[trace_mw], layout=layout_mw)

    # Convertir a JSON para enviar al frontend y renderizar con Plotly.js
    fig_dbm_json = json.dumps(fig_dbm, cls=plotly.utils.PlotlyJSONEncoder)
    fig_mw_json = json.dumps(fig_mw, cls=plotly.utils.PlotlyJSONEncoder)

    return fig_dbm_json, fig_mw_json

def imprimir_resultados_console(resultados):
    print(f"Potencia de la señal inicial (dBm): {resultados['potencia_inicial_dbm']:.2f}\n")
    for d in resultados['detalles_tramos']:
        print(f"Tramo {d['numero']}: Longitud acumulada = {d['longitud_acumulada']:.2f} km, "
              f"Potencia final = {d['potencia_final_dbm']:.2f} dBm, "
              f"Atenuación = {d['atenuacion_tramo']:.2f} dB\n")
    print("Resultados Finales (basado en la simulación):")
    print(f"Potencia de la señal inicial (dBm): {resultados['potencia_inicial_dbm']:.2f}")
    print(f"Potencia de la señal recibida (dBm): {resultados['potencia_final_dbm']:.2f}")
    print(f"Potencia de la señal recibida (mW): {db2lin(resultados['potencia_final_dbm']):.2f}")
    print(f"Atenuación total simulada (dB): {resultados['atenuacion_total']:.2f}")
    print(f"Sensibilidad del receptor (dBm): {resultados['sensibilidad_receptor_dbm']:.2f}")
    if resultados['potencia_final_dbm'] < resultados['sensibilidad_receptor_dbm']:
        print("\nAdvertencia: La potencia de la señal recibida es menor que la sensibilidad del receptor.")

# Si quieres ejecutar manualmente desde consola (opcional)
def main_interactivo():
    valores_default_fibra = {
        'loss_coef': 0.2,
        'att_in': 0,
        'con_in': 0.25,
        'con_out': 0.30,
    }
    import sys
    if sys.version_info.major < 3:
        input_function = raw_input
    else:
        input_function = input

    print("=== Simulación de red óptica con parámetros independientes por tramo ===")
    def solicitar_float(mensaje, valor_por_defecto=None, minimo=None, maximo=None):
        while True:
            try:
                entrada = input_function(f"{mensaje} " + (f"[Por defecto: {valor_por_defecto}]: " if valor_por_defecto is not None else ": "))
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
                valor = int(input_function(mensaje + ": "))
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
        cambiar = input_function("¿Desea cambiar los parámetros de la fibra de este tramo? (s/n): ").lower()
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

    tx_power_dbm = solicitar_float("Introduce la potencia de entrada (dBm)", valor_por_defecto=16.53)
    sensibilidad_receptor_dbm = solicitar_float("Introduce la sensibilidad del receptor (dBm)", valor_por_defecto=-28)
    num_tramos = solicitar_int("Introduce el número de tramos de fibra (1 a 4)", minimo=1, maximo=4)

    tramos_params = []
    for i in range(num_tramos):
        print(f"\nTramo {i+1}:")
        longitud = solicitar_float("Introduce la longitud del tramo (km)", minimo=0.1)
        fibra_parametros = pedir_parametros_tramo(i+1, valores_default_fibra)
        fibra_parametros['length'] = longitud
        tramos_params.append(fibra_parametros)

    resultados = simular_red_por_tramos_detallado(
        tx_power_dbm=tx_power_dbm,
        sensibilidad_receptor_dbm=sensibilidad_receptor_dbm,
        tramos_params=tramos_params
    )
    imprimir_resultados_console(resultados)

 

if __name__ == '__main__':
    main_interactivo()
