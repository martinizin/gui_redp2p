import numpy as np
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, Transceiver
from gnpy.core.utils import watt2dbm, db2lin

def dbm2mw(dbm):
    return 10 ** (dbm / 10)
# Función para obtener la topología de la red
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

# Función para realizar los cálculos de la red óptica
def calcular_red(params):
    """
    Calcula la propagación de la señal en la red óptica basado en los parámetros proporcionados
    
    Args:
        params (dict): Diccionario con los siguientes parámetros:
            - tx_power_dbm: Potencia del transmisor en dBm
            - tx_power_watts: Potencia del transmisor en watts
            - sensitivity_receiver_dbm: Sensibilidad del receptor en dBm
            - fiber_params: Lista de parámetros para cada tramo de fibra
                - loss_coef: Coeficiente de pérdida en dB/km
                - att_in: Atenuación en la entrada (dB)
                - con_in: Pérdida del conector de entrada (dB)
                - con_out: Pérdida del conector de salida (dB)
                - num_stretches: Número de tramos
                - length_stretch: Longitud de cada tramo
    """
    
    # Parámetros del sistema
    f_min = 191.4e12
    f_max = 195.1e12
    spacing = 50e9
    roll_off = 0.15
    tx_osnr = 40
    baud_rate = 32e9
    delta_pdb = 0
    slot_width = spacing

    # Crear información espectral
    si = create_input_spectral_information(f_min, f_max, roll_off, baud_rate, spacing, tx_osnr, params['tx_power_dbm'], slot_width)
    si.signal = si.signal.astype(np.float64)
    
    # Inicializar variables
    power_history_dbm = [params['tx_power_dbm']]
    accumulated_length = [0]
    attenuation_history = []
    accumulated_power_dbm = params['tx_power_dbm']

    # Procesar cada tramo de fibra
    for fiber_param in params['fiber_params']:
        # Parámetros de la fibra para este tramo
        fiber_params = {
            'length': fiber_param['length_stretch'],  # Longitud del tramo
            'loss_coef': fiber_param['loss_coef'],    # Coeficiente de pérdida
            'length_units': 'km',
            'att_in': fiber_param['att_in'],
            'con_in': fiber_param['con_in'],
            'con_out': fiber_param['con_out'],
            'pmd_coef': 0.1,
            'dispersion': 16.5,
            'gamma': 1.2,
            'effective_area': 80e-12,
            'core_radius': 4.2e-6,
            'n1': 1.468,
            'n2': 2.6e-20
        }
        
        fiber = Fiber(uid="Fiber", params=fiber_params)
        power_before_fiber = watt2dbm(np.sum(si.signal / 1000))
        fiber.ref_pch_in_dbm = power_before_fiber

        # Calcular atenuación para este tramo
        attenuation = fiber_params['loss_coef'] * fiber_params['length']
        attenuation += fiber_params['att_in'] + fiber_params['con_in'] + fiber_params['con_out']
        
        accumulated_power_dbm -= attenuation
        power_history_dbm.append(accumulated_power_dbm)
        accumulated_length.append(accumulated_length[-1] + fiber_params['length'])
        attenuation_history.append(attenuation)

    # Convertir potencia final a mW
    final_power_mw = db2lin(accumulated_power_dbm)

    return {
        "initial_power": params['tx_power_dbm'],
        "final_power": accumulated_power_dbm,
        "final_power_mw": final_power_mw,
        "attenuation": sum(attenuation_history),
        "receiver_sensitivity": params['sensitivity_receiver_dbm'],
        "longitud_acumulada": accumulated_length,
        "power_history_dbm": power_history_dbm,
        "power_history_linear": [db2lin(dbm) for dbm in power_history_dbm],
        "attenuation_history": attenuation_history
    }


# Función para obtener los datos de la red
