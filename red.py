import numpy as np
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, Transceiver
from gnpy.core.utils import watt2dbm, db2lin

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
def calcular_red(tx_power, receiver_sensitivity, num_segments, segment_lengths):
    f_min = 191.4e12
    f_max = 195.1e12
    spacing = 50e9
    roll_off = 0.15
    tx_osnr = 40
    baud_rate = 32e9
    delta_pdb = 0
    slot_width = spacing

    si = create_input_spectral_information(f_min, f_max, roll_off, baud_rate, spacing, tx_osnr, tx_power, slot_width)
    si.signal = si.signal.astype(np.float64)
    
    num_channels = int(np.floor((f_max - f_min) / spacing))
    power_chanel = 10**(tx_power/10)
    total_input_power_w = num_channels * power_chanel
    total_input_power_dbm = watt2dbm(total_input_power_w / 1000)

    # Parámetros de la fibra
    fiber_params = {
        'length': sum(segment_lengths),  # Longitud total de la fibra (en km)
        'loss_coef': 0.2,               # Coeficiente de pérdida en dB/km
        'length_units': 'km',           # Unidades de longitud (en km)
        'att_in': 0,                    # Atenuación en la entrada (dB)
        'con_in': 0.25,                 # Pérdida del conector de entrada (dB)
        'con_out': 0.30,                # Pérdida del conector de salida (dB)
        'pmd_coef': 0.1,                # PMD
        'dispersion': 16.5,             # Dispersión cromática
        'gamma': 1.2,                   # Coeficiente no lineal
        'effective_area': 80e-12,       # Área efectiva
        'core_radius': 4.2e-6,          # Radio del núcleo en m
        'n1': 1.468,                    # Índice de refracción del núcleo
        'n2': 2.6e-20                   # Coeficiente no lineal
    }
    
    fiber = Fiber(uid="Fiber1", params=fiber_params)
    power_before_fiber = watt2dbm(np.sum(si.signal / 1000))
    fiber.ref_pch_in_dbm = power_before_fiber

    power_history_dbm = [tx_power]
    accumulated_length = [0]
    attenuation_history = []
    accumulated_power_dbm = tx_power

    for i in range(num_segments):
        segment_length = segment_lengths[i]
        if segment_length <= 0:
            continue 
        attenuation = fiber_params['loss_coef'] * segment_length
        accumulated_power_dbm -= attenuation
        power_history_dbm.append(accumulated_power_dbm)
        accumulated_length.append(accumulated_length[-1] + segment_length)
        attenuation_history.append(accumulated_power_dbm - tx_power)
        
    final_power_mw = db2lin(accumulated_power_dbm)

    return {
        "initial_power": tx_power,
        "final_power": accumulated_power_dbm,
        "final_power_mw": final_power_mw,
        "attenuation": sum(attenuation_history),
        "receiver_sensitivity": receiver_sensitivity,
        "longitud_acumulada": accumulated_length,
        "power_history_dbm": power_history_dbm,
        "power_history_linear": power_history_dbm,  # Se debe ajustar a la potencia lineal
        "attenuation_history": attenuation_history
    }
