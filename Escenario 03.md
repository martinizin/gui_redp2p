# Documentación del Escenario 03

## Resumen
El **Escenario 03** es un sistema avanzado de **planificación y optimización de rutas** en redes ópticas complejas. Utiliza algoritmos de optimización de rutas, gestión inteligente de topologías y cálculos profesionales con **gnpy** para encontrar las mejores rutas entre nodos en redes de múltiples puntos. Este escenario está diseñado para operadores de red y planificadores que requieren análisis de rutas óptimas en topologías extensas.

## ¿Qué se desarrolló?

### Funcionalidades de Planificación de Red:

#### 1. **Gestión Inteligente de Topologías**
- Carga automática de topologías por defecto y subidas por usuario
- Sistema de archivos temporales para pruebas sin afectar configuraciones permanentes
- Validación automática de requisitos para redes bidireccionales
- Limpieza automática de archivos temporales al recargar la página

#### 2. **Cálculo de Rutas Óptimas**
- Algoritmo de búsqueda de rutas múltiples entre cualquier par de nodos
- Optimización por diferentes criterios:
  - **Por OSNR**: Mejor calidad de señal
  - **Por Distancia**: Menor longitud de ruta
- Cálculo simultáneo de múltiples rutas alternativas
- Ranking automático de rutas encontradas

#### 3. **Simulación Profesional con gnpy**
- Motor de cálculo basado en `transmission_simulation` de gnpy
- Análisis completo de métricas:
  - **SNR en 0.1nm** y **ancho de banda del canal**
  - **OSNR_ASE en 0.1nm** y **ancho de banda del canal**
  - **Distancia total** de la ruta
- Simulación de equipos reales desde configuraciones de fabricante

#### 4. **Validación de Redes Bidireccionales**
- Análisis automático de conectividad bidireccional
- Detección de patrones de nomenclatura de fibras (A→B, B→A)
- Validación permisiva que permite cálculos con advertencias
- Recomendaciones de mejores prácticas para topologías robustas

#### 5. **Interfaz de Configuración Avanzada**
- Editor de parámetros ópticos del sistema (frecuencias, potencias, márgenes)
- Configuración de transceivers disponibles
- Parámetros SI (Spectral Information) configurables
- Integración con Google Maps API para visualización geográfica

### Componentes Técnicos Desarrollados:

#### Gestión de Archivos:
- `get_default_topology_files()`: Gestión de topologías permanentes
- `get_temp_topology_files()`: Manejo de uploads temporales
- `upload_topology_file()`: Carga segura con validación
- `clean_temp_topologies()`: Limpieza automática al iniciar

#### Cálculo de Rutas:
- `calculate_routes()`: Motor principal de optimización de rutas
- Integración con `nx.shortest_simple_paths()` para búsqueda eficiente
- `designed_network()` y `transmission_simulation()` de gnpy
- Ordenamiento automático por criterios de optimización

#### Validación de Topologías:
- `validate_topology_requirements()`: Validación flexible con niveles de severidad
- `validate_bidirectional_connections()`: Análisis de conectividad bidireccional
- `extract_logical_roadm_connections()`: Extracción de conexiones lógicas
- Sistema de errores críticos vs advertencias

#### Análisis de Resultados:
- `tipo()`: Clasificación automática de tipos de nodos
- Formateo científico de parámetros (`format_scientific_notation()`)
- Captura de mensajes de consola para depuración
- Métricas detalladas por ruta encontrada

## Parámetros de Configuración del Sistema

### Parámetros Espectrales (SI - Spectral Information):
- **Frecuencia mínima**: 191.3 THz (inicio banda C)
- **Frecuencia máxima**: 196.1 THz (fin banda C extendida)
- **Espaciado de canales**: 50 GHz (estándar ITU-T)
- **Velocidad de baudios**: 32 GBaud (alta velocidad)
- **Roll-off**: 0.15 (factor de forma Nyquist)
- **OSNR de transmisión**: 35 dB (valor típico)
- **Márgenes del sistema**: 2 dB (seguridad operacional)

### Configuración de Potencia:
- **Potencia por defecto**: 2 dBm (por canal)
- **Potencia de transmisión**: 0 dBm (configurable)
- **Rango de potencia**: [0, 0, 1] dB (control dinámico)

### Criterios de Optimización:
1. **Por OSNR** (por defecto): Maximizar calidad de señal
2. **Por Distancia**: Minimizar longitud de ruta

## Topologías Soportadas

### Archivos por Defecto Incluidos:
- `CORONET_Global_Topology.json`: Red global de investigación
- `Sweden_OpenROADMv4_example_network.json`: Red sueca OpenROADM
- `topologiaEC.json`: Topología ecuatoriana

### Requisitos de Topología:

#### Requisitos Críticos (bloquean cálculos):
- **Mínimo 2 Transceivers**: Para definir origen y destino
- **Al menos 1 conexión**: Entre elementos para formar red

#### Mejores Prácticas (advertencias):
- **Mínimo 3 Transceivers**: Para redes robustas
- **Mínimo 2 ROADMs**: Para topologías complejas
- **Presencia de EDFAs**: Para enlaces largos
- **Conectividad bidireccional**: Para redundancia

### Estructura de Topología Requerida:
```json
{
  "elements": [
    {
      "uid": "trx_madrid",
      "type": "Transceiver",
      "metadata": {
        "location": {
          "latitude": 40.4168,
          "longitude": -3.7038
        }
      }
    },
    {
      "uid": "roadm_barcelona", 
      "type": "Roadm",
      "metadata": {
        "location": {
          "latitude": 41.3851,
          "longitude": 2.1734
        }
      }
    },
    {
      "uid": "fiber_madrid_barcelona",
      "type": "Fiber",
      "params": {
        "length": 620000,        # metros
        "loss_coef": 0.2,        # dB/km
        "con_in": 0.5,           # dB
        "con_out": 0.5           # dB
      }
    }
  ],
  "connections": [
    {
      "from_node": "trx_madrid",
      "to_node": "roadm_barcelona"
    }
  ]
}
```

## Requisitos para Ejecutar (si se clona el proyecto)

### Dependencias Esenciales:
```python
gnpy>=2.4.0              # Motor de simulación óptica
networkx>=2.6.0          # Algoritmos de grafos y rutas
flask>=2.0.0             # Framework web
numpy>=1.19.0            # Cálculos numéricos
pathlib                  # Manejo de rutas de archivos
python-dotenv>=0.19.0    # Variables de entorno
```

### Instalación Completa:
```bash
# Clonar repositorio gnpy (recomendado)
git clone https://github.com/Telecominfraproject/oopt-gnpy.git
cd oopt-gnpy
pip install -e .

# Instalar dependencias adicionales
pip install networkx flask numpy python-dotenv
```

### Variables de Entorno (.env):
```bash
# Crear archivo .env en la raíz del proyecto
EQPT_DIR=data
TOPOLOGY_DIR=topologias
MAPS_API_KEY=your_google_maps_api_key_here
```

### Estructura de Directorios:
```
proyecto/
├── .env                          # Variables de entorno
├── data/
│   └── eqpt_configv1.json       # Configuración de equipos
├── topologias/
│   ├── CORONET_Global_Topology.json
│   ├── Sweden_OpenROADMv4_example_network.json
│   └── temp_uploads/            # Dir temporal (se crea automáticamente)
└── scenario03.py                # Código principal
```

## Ejecución y Uso

### 1. **Configuración Inicial**:
```python
from scenario03 import handle_scenario03

# Inicializar escenario (limpia archivos temporales)
response = handle_scenario03()
print("Escenario 03 iniciado correctamente")
```

### 2. **Carga de Topología**:
```python
from scenario03 import get_topology_data

# Cargar topología específica
topology_data = get_topology_data('CORONET_Global_Topology.json')

# Verificar validación
if topology_data['validation']['valid']:
    print("Topología válida para cálculos")
else:
    print(f"Advertencias: {topology_data['validation']['message']}")
```

### 3. **Cálculo de Rutas**:
```python
from scenario03 import calculate_routes
import json

# Parámetros de cálculo
params = {
    'topology_filename': 'CORONET_Global_Topology.json',
    'source_node': 'trx_atlanta',
    'destination_node': 'trx_seattle', 
    'number_of_routes': 3,
    'calculation_criteria': 'osnr',  # o 'distance'
    'optical_parameters': {
        'f_min': 191.3e12,
        'f_max': 196.1e12,
        'spacing': 50e9,
        'baud_rate': 32e9,
        'tx_osnr': 35,
        'power_dbm': 2
    }
}

# Ejecutar cálculo
results = calculate_routes()  # Usa request.get_json() en contexto Flask

if results['success']:
    print(f"Encontradas {results['num_routes_found']} rutas")
    
    for route in results['routes']:
        print(f"Ruta {route['ruta_num']}:")
        print(f"  SNR: {route['snr_01nm']:.2f} dB")
        print(f"  OSNR: {route['osnr_01nm']:.2f} dB") 
        print(f"  Distancia: {route['distancia_total_km']:.1f} km")
```

### 4. **Análisis de Resultados**:
```python
# Acceder a métricas detalladas
best_route = results['routes'][0]  # Mejor ruta (índice 0)

print(f"Mejor ruta: {best_route['receptor_uid']}")
print(f"Nodos en la ruta:")
for node in best_route['ruta_nodos']:
    print(f"  {node['idx']}: {node['uid']} ({node['tipo']})")

print(f"Fibras utilizadas:")
for fiber in best_route['fibras']:
    print(f"  {fiber['uid']}: {fiber['length_km']:.1f} km")
```

## Métricas y Análisis de Resultados

### Métricas por Ruta:
- **SNR en 0.1nm**: Relación señal-ruido en ancho de banda de referencia
- **SNR en Bw**: SNR en ancho de banda del canal
- **OSNR_ASE en 0.1nm**: OSNR considerando solo ruido ASE
- **OSNR_ASE en Bw**: OSNR en ancho de banda del canal
- **Distancia Total**: Suma de longitudes de fibras en la ruta
- **Lista de Nodos**: Secuencia completa de elementos atravesados

### Criterios de Optimización:

#### Por OSNR (recomendado):
1. Maximizar SNR en 0.1nm
2. Minimizar distancia (como criterio secundario)
3. Ideal para garantizar calidad de señal

#### Por Distancia:
1. Minimizar distancia total
2. Maximizar SNR (como criterio secundario)  
3. Ideal para minimizar latencia y costos

### Información de Depuración:
- **Mensajes de Consola**: Captura completa de salidas de gnpy
- **Advertencias de Topología**: Problemas no críticos detectados
- **Errores de Simulación**: Detalles de rutas que fallaron
- **Parámetros Aplicados**: Configuración utilizada en los cálculos

## Ventajas del Escenario 03

- ✅ **Optimización Multi-Ruta**: Encuentra mejores alternativas automáticamente
- ✅ **Criterios Flexibles**: OSNR vs Distancia según necesidades
- ✅ **Topologías Reales**: Soporta redes complejas del mundo real
- ✅ **Validación Inteligente**: Permite cálculos con advertencias informativas
- ✅ **Gestión de Archivos**: Sistema seguro de uploads temporales
- ✅ **Métricas Completas**: Análisis profesional con gnpy
- ✅ **Escalabilidad**: Eficiente para redes grandes (cientos de nodos)
- ✅ **Depuración Avanzada**: Información detallada para troubleshooting

## Limitaciones y Consideraciones

### Limitaciones Técnicas:
- ⚠️ **Dependencia de gnpy**: Requiere instalación y configuración correcta
- ⚠️ **Complejidad de Topología**: Requiere archivos JSON bien formateados
- ⚠️ **Recursos Computacionales**: Cálculos intensivos para redes grandes

### Consideraciones Operacionales:
- **Archivos Temporales**: Se eliminan al recargar la página
- **Validación Permisiva**: Permite cálculos con advertencias
- **Configuración de Equipos**: Requiere archivo `eqpt_configv1.json` válido


Este escenario representa la **una herramienta** para planificación de redes ópticas, combinando algoritmos de optimización de rutas con simulaciones mediante gnpy.
