import numpy as np
import math
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, Transceiver
from gnpy.core.utils import watt2dbm, db2lin, dbm2watt

# Función para obtener la topología de la red
def obtener_topologia_datos():
    # Nodos: el nodo de la fibra se elimina ya que no se quiere mostrar
    nodos = [
        {"id": 1, "label": "Transmisor", "x": 100, "y": 100},  # Nodo 1: Transmisor
        {"id": 3, "label": "Receptor", "x": 500, "y": 100}     # Nodo 3: Receptor
    ]
    
    # Enlaces: Definir los enlaces que conectan los nodos (Emisor -> Fibra -> Receptor)
    enlaces = [
        {"from": 1, "to": 3, "label": "", "length": 5}  # Enlace entre el Emisor y el Receptor, representando la fibra
    ]
    
    return nodos, enlaces


# Función para realizar los cálculos de la red óptica
def calcular_red(fiber_length=100):
    # Parámetros de la simulación
    f_min = 191.4e12  # Frecuencia mínima 
    f_max = 195.1e12  # Frecuencia máxima 
    spacing = 50e9    # Espaciado entre canales 
    roll_off = 0.15   # Roll-off del filtro
    tx_osnr = 40      # OSNR del transmisor (dB)
    tx_power = 6      # Potencia del transmisor (dBm)
    baud_rate = 32e9  # Tasa de baudios 
    delta_pdb = 0     # Delta de potencia (dB)
    slot_width = spacing  # Ancho de slot 

    # Crear el objeto SpectralInformation
    si = create_input_spectral_information(f_min, f_max, roll_off, baud_rate, spacing, tx_osnr, tx_power, slot_width)
    si.signal = si.signal.astype(np.float64)

    # Calcular número de canales
    num_channels = int(np.floor((f_max - f_min) / spacing))

    # Calcular potencia por canal en mW
    power_chanel = 10**(tx_power/10)

    # Calcular potencia total de entrada
    total_input_power_w = num_channels*power_chanel
    total_input_power_dbm = watt2dbm(total_input_power_w/1000)

    # Parámetros de la fibra
    fiber_params = {
        'length': fiber_length,         # Longitud de la fibra en km
        'loss_coef': 0.2,     # Coeficiente de pérdida en dB/km
        'length_units': 'km', # Unidades de longitud
        'att_in': 0,          # Atenuación en la entrada (dB)
        'con_in': 0.25,       # Conector de entrada (dB)
        'con_out': 0.30,      # Conector de salida (dB)
        'pmd_coef': 0.1,      # PMD 
        'dispersion': 16.5,   # Dispersión cromática 
        'gamma': 1.2,         # Coeficiente no lineal 
        'effective_area': 80e-12,  # Área efectiva
        'core_radius': 4.2e-6,     # Radio del núcleo en m
        'n1': 1.468,               # Índice de refracción del núcleo
        'n2': 2.6e-20              # Coeficiente no lineal
    }

    # Crear una fibra óptica
    fiber = Fiber(uid="Fiber1", params=fiber_params)

    # Calcular y mostrar la potencia antes de la fibra
    power_before_fiber = watt2dbm(np.sum(si.signal/1000))

    # IMPORTANTE: Establecer la potencia de referencia en la fibra
    fiber.ref_pch_in_dbm = power_before_fiber

    # Propagar la señal a través de la fibra
    si_after_fiber = fiber(si)

    # Calcular la potencia después de la fibra
    power_after_fiber = watt2dbm(np.sum(si_after_fiber.signal/1000))

    # Crear un receptor
    trx = Transceiver(uid="Receiver")
    si_received = trx(si_after_fiber)

    # Calcular la potencia de la señal recibida en dBm
    signal_power_watts = np.sum(si_received.signal/1000)
    signal_power_dbm = watt2dbm(signal_power_watts)

    # Calcular la atenuación total esperada
    expec = total_input_power_dbm
    fiber_att = fiber_params['loss_coef'] * fiber_params['length'] + fiber_params['con_in'] + fiber_params['con_out']
    expected_power_after_fiber = expec - fiber_att

    # Calcular el OSNR en diferentes puntos del sistema
    def calculate_osnr(signal_info):
        signal_power_total = np.sum(signal_info.signal)/1000  # mW a W
        noise_power_total = np.sum(signal_info.nli + signal_info.ase)/1000  # mW a W
        if noise_power_total == 0:
            return tx_osnr 
        osnr_db = 10 * np.log10(signal_power_total / noise_power_total)
        return osnr_db

    # Inicializar valores de OSNR
    osnr_values = []
    osnr_tx = tx_osnr
    osnr_values.append(osnr_tx)
    osnr_before_fiber = osnr_tx
    osnr_values.append(osnr_before_fiber)
    osnr_after_fiber = calculate_osnr(si_after_fiber)
    osnr_values.append(osnr_after_fiber)
    osnr_rx = calculate_osnr(si_received)
    osnr_values.append(osnr_rx)

    # Devolver los resultados de la red
    return {
        "potencia_antes_fibra": power_before_fiber,
        "potencia_despues_fibra": power_after_fiber,
        "potencia_recibida": signal_power_dbm,
        "osnr_tx": osnr_tx,
        "osnr_before_fiber": osnr_before_fiber,
        "osnr_after_fiber": osnr_after_fiber,
        "osnr_rx": osnr_rx,
        "nodos": obtener_topologia_datos()[0],  # Nodos de la red
        "enlaces": obtener_topologia_datos()[1] # Enlaces de la red
    }
