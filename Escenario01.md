# Documentación del Escenario 01

## Resumen
El **Escenario 01** es un simulador de propagación de señales ópticas que modela la atenuación de potencia a través de segmentos de fibra óptica. Este escenario se enfoca en el análisis detallado de pérdidas de potencia y la verificación de viabilidad de enlaces punto a punto.

## ¿Qué se desarrolló?

### Funcionalidades principales:
1. **Simulación de Propagación de Señal**: Modela la atenuación de señales ópticas através de múltiples tramos de fibra
2. **Cálculo de Pérdidas**: Considera pérdidas por:
   - Coeficiente de atenuación de la fibra (dB/km)
   - Conectores de entrada y salida
   - Atenuaciones internas adicionales
3. **Visualización Interactiva**: Genera gráficos tanto en escala logarítmica (dBm) como lineal (mW)
4. **Interfaz Web**: Integración completa con la aplicación Flask para uso web
5. **Entrada de Parámetros**: Sistema interactivo para configurar parámetros de cada tramo

### Componentes técnicos desarrollados:

#### Funciones de Conversión:
- `dbm2mw()`: Conversión de dBm a miliWatts
- Funciones de entrada robusta (`solicitar_float()`, `solicitar_int()`)

#### Simulación Principal:
- `simular_red_por_tramos_detallado()`: Motor de simulación que procesa cada segmento de fibra
- Procesamiento por segmentos de 5km para mayor precisión
- Cálculo acumulativo de pérdidas

#### Visualización:
- `graficar_potencia()`: Gráficos matplotlib para análisis detallado
- `graficar_potencia_plotly_linear()` y `graficar_potencia_plotly_dbm()`: Visualización web interactiva
- Comparación automática con sensibilidad del receptor

#### Integración Web:
- `calcular_red()`: Función principal para cálculos desde la interfaz web
- `obtener_topologia_datos()`: Generación de datos de topología simple

## Parámetros de Configuración

### Parámetros del Transmisor:
- **Potencia de transmisión**: 16.53 dBm (por defecto)
- **OSNR de transmisión**: 40 dB

### Parámetros del Receptor:
- **Sensibilidad**: -28 dBm (por defecto)

### Parámetros de Fibra (por tramo):
- **Coeficiente de pérdida**: 0.2 dB/km (típico para fibra monomodo)
- **Pérdida conector entrada**: 0.25 dB
- **Pérdida conector salida**: 0.30 dB
- **Atenuación interna**: 0 dB (configurable)
- **Longitud del tramo**: 5 km (configurable)

## Casos de Uso

### 1. Análisis de Enlaces Punto a Punto
- Verificación de viabilidad de enlaces directos
- Cálculo de márgenes de potencia
- Optimización de parámetros de transmisión

### 2. Diseño de Redes Simples
- Planificación de enlaces de fibra óptica
- Evaluación de diferentes configuraciones de fibra
- Análisis de sensibilidad a parámetros

### 3. Educación y Entrenamiento
- Comprensión de principios básicos de propagación óptica
- Visualización de efectos de atenuación
- Experimentación con diferentes parámetros

## Requisitos para Ejecutar (si se clona el repositorio)

### Dependencias Python:
```python
numpy>=1.19.0
matplotlib>=3.3.0
plotly>=5.0.0
gnpy>=2.4.0  # Para funciones de utilidad
flask>=2.0.0  # Para integración web
```

### Instalación:
```bash
pip install numpy matplotlib plotly gnpy flask
```

### Archivos Requeridos:
- `scenario01.py`: Código principal del escenario
- `templates/index.html`: Interfaz web (si se usa integración web)
- `static/`: Archivos CSS y JavaScript para la interfaz

### Estructura de Datos de Entrada:
```python
# Ejemplo de parámetros para calcular_red()
params = {
    'tx_power_dbm': 16.53,
    'sensitivity_receiver_dbm': -28,
    'fiber_params': [
        {
            'length_stretch': 50,      # km
            'loss_coef': 0.2,          # dB/km
            'att_in': 0.0,             # dB
            'con_in': 0.25,            # dB
            'con_out': 0.30            # dB
        }
        # ... más tramos según sea necesario
    ]
}
```

## Ejecución

### Modo Interactivo (Consola):
```python
python scenario01.py
# Sigue las instrucciones para ingresar parámetros
```

### Modo Web:
```python
from scenario01 import calcular_red

# Ejecutar cálculos
resultados = calcular_red(params)

# Los resultados incluyen:
# - Historial completo de potencia
# - Gráficos Plotly listos para web
# - Verificación automática de viabilidad
```

### Resultados Obtenidos:
- **Gráfico de Potencia vs Distancia** (dBm y lineal)
- **Análisis de Viabilidad**: Comparación con sensibilidad del receptor
- **Detalles por Tramo**: Atenuación y potencia final de cada segmento
- **Métricas del Sistema**: Atenuación total, longitud total, márgenes

## Ventajas del Escenario 01
- ✅ **Simplicidad**: Fácil de entender y usar
- ✅ **Precisión**: Modelado detallado de pérdidas por tramo
- ✅ **Visualización**: Gráficos claros y comprensibles
- ✅ **Flexibilidad**: Configurable para diferentes tipos de fibra y conectores
- ✅ **Educativo**: Excelente para aprender principios básicos

## Limitaciones
- ⚠️ **Alcance**: Limitado a análisis de potencia (no incluye OSNR detallado)
- ⚠️ **Complejidad**: No maneja redes complejas con múltiples rutas
- ⚠️ **Elementos**: No incluye amplificadores ópticos (EDFAs)

El Escenario 01 es ideal para **análisis básicos de viabilidad de enlaces** y como **herramienta educativa** para comprender los fundamentos de la propagación en fibra óptica.
