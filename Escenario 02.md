# Documentación del Escenario 02

## Resumen
El **Escenario 02** es un simulador avanzado de redes ópticas que utiliza la librería profesional **gnpy** (Gaussian Noise model in Python) para realizar cálculos detallados de OSNR, análisis de ruido ASE y NLI, y simulaciones de propagación en redes punto a punto complejas. Este escenario representa el estado del arte en simulación de redes ópticas WDM (Wavelength Division Multiplexing).

## ¿Qué se desarrolló?

### Funcionalidades Avanzadas:

#### 1. **Simulación Profesional con gnpy**
- Motor de cálculo basado en la librería gnpy de Telecom Infra Project
- Modelado gaussiano de ruido para precisión industrial
- Simulación de elementos reales: Transceivers, EDFAs, Fibras
- Cálculos de OSNR en diferentes anchos de banda (Bw y 0.1nm)

#### 2. **Gestión de Topologías Complejas**
- Carga y procesamiento de archivos JSON de topología
- Visualización automática de redes (mapas geográficos o layouts horizontales)
- Detección automática de coordenadas geográficas
- Manejo de cadenas de fibras y elementos intermedios

#### 3. **Análisis Multi-Canal WDM**
- Simulación de múltiples canales ópticos simultáneos
- Cálculo automático de potencia por canal desde potencia total
- Análisis de efectos no lineales (NLI - Nonlinear Interference)
- Ruido ASE (Amplified Spontaneous Emission) detallado

#### 4. **Visualización Interactiva Avanzada**
- Topologías en mapas geográficos con OpenStreetMap
- Layouts horizontales para redes sin coordenadas
- Tooltips detallados con parámetros de equipos
- Colores y símbolos diferenciados por tipo de elemento

#### 5. **Gestión de Parámetros de Red**
- Editor interactivo de parámetros EDFA (ganancia, NF, potencia máxima)
- Configuración de transceivers (potencia, sensibilidad, OSNR requerido)
- Parámetros de fibra automáticos desde topología
- Actualización dinámica de configuraciones

### Componentes Técnicos Desarrollados:

#### Motor de Simulación:
- `calculate_scenario02_network()`: Función principal de cálculo con gnpy
- Integración completa con objetos gnpy (Transceiver, Fiber, Edfa)
- Simulación etapa por etapa del enlace óptico
- Cálculo de OSNR clásico paralelo para verificación

#### Procesamiento de Topologías:
- `process_scenario02_data()`: Carga y mejora de archivos de topología
- `enhance_elements_with_parameters()`: Enriquecimiento automático de parámetros
- `validate_topology_requirements()`: Validación de requisitos mínimos
- `order_elements_by_topology()`: Ordenamiento por conexiones reales

#### Visualización de Redes:
- `_create_map_plot()`: Visualización geográfica con mapbox
- `_create_horizontal_plot()`: Layout lineal para análisis
- `detect_coordinate_overlaps()`: Manejo de elementos co-ubicados
- `apply_coordinate_offsets()`: Separación visual automática

#### Análisis de Resultados:
- `generate_scenario02_plots()`: Gráficos de evolución de señal
- Gráficos separados: Potencia de señal, Potencia ASE, OSNR vs Distancia
- Formateo científico automático de resultados
- Verificación de criterios operacionales

## Parámetros de Configuración Avanzada

### Parámetros del Sistema WDM:
- **Frecuencia mínima**: 191.3 THz (banda C)
- **Frecuencia máxima**: 195.1 THz 
- **Espaciado de canales**: 50 GHz (compatible con ITU-T)
- **Velocidad de baudios**: 32 GBaud
- **Roll-off**: 0.15 (factor de forma espectral)
- **Ancho de banda de referencia**: 12.5 GHz

### Parámetros de Transceptores:
#### Transmisor (Source):
- **Potencia total de entrada**: Configurable (dBm)
- **OSNR de transmisión**: 40 dB (valor típico)
- **Distribución automática**: Potencia dividida entre canales

#### Receptor (Destination):
- **Sensibilidad**: Configurable (dBm)
- **OSNR requerido**: 15 dB (umbral operacional)
- **Verificación dual**: Potencia Y OSNR deben cumplirse

### Parámetros EDFA Avanzados:
- **Ganancia objetivo**: 20 dB (configurable por usuario)
- **Factor de ruido (NF)**: 6 dB (editable en tiempo real)
- **Ganancia máxima plana**: 26 dB (desde configuración de equipos)
- **Potencia de saturación**: 23 dBm (límite físico)

## Casos de Uso Profesionales

### 1. **Diseño de Enlaces Metropolitanos**
- Verificación de viabilidad de enlaces de 80-200 km
- Optimización de configuraciones EDFA
- Análisis de márgenes operacionales

### 2. **Planificación de Redes WDM**
- Cálculo de capacidad de canales ópticos
- Optimización de potencia por canal
- Análisis de efectos no lineales

### 3. **Validación de Equipos**
- Verificación de especificaciones de EDFAs
- Comparación de diferentes tipos de fibra
- Análisis de sensibilidad del receptor

### 4. **Estudios de Factibilidad**
- Evaluación técnica de proyectos de fibra óptica
- Análisis costo-beneficio de configuraciones
- Informes técnicos profesionales

## Requisitos para Ejecutar

### Dependencias Críticas:
```python
gnpy>=2.4.0          # Librería principal (OBLIGATORIA)
numpy>=1.19.0        # Cálculos numéricos
plotly>=5.0.0        # Visualización interactiva
flask>=2.0.0         # Framework web
pathlib              # Manejo de rutas
json                 # Procesamiento de topologías
```

### Instalación de gnpy:
```bash
# Opción 1: Instalación básica
pip install gnpy

# Opción 2: Instalación desde fuente (recomendada)
git clone https://github.com/Telecominfraproject/oopt-gnpy.git
cd oopt-gnpy
pip install -e .
```

### Archivos de Configuración Requeridos:
```
data/
├── eqpt_final.json          # Configuración de equipos EDFA
├── enlace_WDM.json          # Parámetros de enlace (opcional)
└── meshTopologyExampleV2.json   # Topología de ejemplo

topologias/
├── CORONET_Global_Topology.json    # Ejemplos de red
├── Sweden_OpenROADMv4_example_network.json
└── topologiaEC.json
```

### Estructura de Archivo de Topología:
```json
{
  "elements": [
    {
      "uid": "site_a",
      "type": "Transceiver",
      "metadata": {
        "latitude": 40.7128,
        "longitude": -74.0060
      }
    },
    {
      "uid": "fiber_a_to_b",
      "type": "Fiber",
      "params": {
        "length": 80000,        # metros
        "loss_coef": 0.2,       # dB/km
        "con_in": 0.5,          # dB
        "con_out": 0.5          # dB
      }
    },
    {
      "uid": "edfa_1",
      "type": "Edfa",
      "type_variety": "std_medium_gain",
      "operational": {
        "gain_target": 20,      # dB
        "tilt_target": 0,       # dB
        "out_voa": 0            # dB
      }
    }
  ],
  "connections": [
    {
      "from_node": "site_a",
      "to_node": "fiber_a_to_b"
    }
  ]
}
```

## Ejecución Avanzada

### 1. **Carga de Topología**:
```python
from scenario02 import process_scenario02_data

# Cargar archivo de topología
with open('topology.json', 'rb') as f:
    result = process_scenario02_data(f)
    
if 'error' not in result:
    topology_data = result['enhanced_data']
    print("Topología cargada exitosamente")
```

### 2. **Configuración de Parámetros**:
```python
# Ejemplo de actualización de parámetros EDFA
from scenario02 import update_scenario02_parameters

# Actualizar ganancia de EDFA
params = {
    'element_uid': 'edfa_1',
    'parameter_name': 'gain_target',
    'new_value': 25.0,
    'topology_data': topology_data
}
```

### 3. **Ejecución de Cálculos**:
```python
from scenario02 import calculate_scenario02

# Ejecutar simulación completa
calculation_params = {
    'topology_data': topology_data
}

results = calculate_scenario02_network(calculation_params)

if results['success']:
    print(f"Potencia final: {results['final_results']['final_power_dbm']:.2f} dBm")
    print(f"OSNR final: {results['final_results']['final_osnr_bw']} dB")
    print(f"Circuito operacional: {results['final_results']['circuit_operational']}")
```

### 4. **Análisis de Resultados**:
```python
# Acceder a resultados detallados
stages = results['stages']
for stage in stages:
    print(f"{stage['name']}: {stage['power_dbm']:.2f} dBm, OSNR: {stage['osnr_bw']} dB")

# Visualizar gráficos
plots = results['plots']  # Contiene gráficos Plotly listos para mostrar
```

## Métricas y Resultados

### Resultados por Etapa:
- **Distancia acumulada** en cada punto
- **Potencia de señal** (total y por canal)
- **OSNR en ancho de banda del canal** (OSNR_bw)
- **OSNR en 0.1nm** (OSNR_01nm) 
- **OSNR clásico paralelo** (para verificación)

### Verificaciones Operacionales:
1. **Condición de Potencia**: P_recibida ≥ Sensibilidad
2. **Condición de OSNR**: OSNR_clásico ≥ OSNR_requerido
3. **Estado del Circuito**: Ambas condiciones deben cumplirse

### Gráficos Generados:
- **Evolución de Potencia de Señal** vs Distancia
- **Evolución de Potencia ASE** vs Distancia  
- **Evolución de OSNR** vs Distancia
- Formato escalonado para mostrar efectos de amplificación

## Ventajas del Escenario 02

- ✅ **Precisión Industrial**: Usa gnpy, estándar de la industria
- ✅ **Análisis Completo**: OSNR, ASE, NLI, efectos no lineales
- ✅ **Topologías Reales**: Soporta redes complejas del mundo real
- ✅ **Visualización Profesional**: Mapas geográficos y layouts técnicos
- ✅ **Configuración Flexible**: Parámetros editables en tiempo real
- ✅ **Validación Dual**: Potencia Y calidad de señal (OSNR)
- ✅ **Multi-Canal**: Análisis WDM completo

## Aplicaciones Profesionales

El Escenario 02 es ideal para:
- **Ingenieros de Redes Ópticas**: Diseño y validación de enlaces
- **Consultores de Telecomunicaciones**: Estudios de factibilidad  
- **Operadores de Red**: Planificación de expansiones
- **Proveedores de Equipos**: Validación de especificaciones
- **Instituciones Educativas**: Enseñanza de conceptos avanzados

Este escenario representa la **herramienta más avanzada** de la suite, proporcionando cálculos de **calidad profesional** para el diseño de redes ópticas modernas.