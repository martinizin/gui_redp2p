# 📡 GUI Red P2P - Simulación de Redes Ópticas

[![Docker](https://img.shields.io/badge/Docker-Available-blue?logo=docker)](https://hub.docker.com/r/martiniziin/proyectognpy)
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-red?logo=flask)](https://flask.palletsprojects.com/)
[![gnpy](https://img.shields.io/badge/gnpy-Network%20Simulation-orange)](https://github.com/Telecominfraproject/oopt-gnpy)

Aplicación web interactiva para la simulación y análisis de redes ópticas, desarrollada con Flask y gnpy. Incluye visualización en Google Maps y análisis avanzado de topologías de red.

## 🚀 Inicio Rápido con Docker

### Prerrequisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop) instalado
- Mínimo 4GB RAM disponible
- Puerto 5000  (sugerencia) libre o cualquier otro puerto

### Ejecutar la Aplicación

```bash
# 1. Descargar la imagen desde Docker Hub
docker pull martiniziin/proyectognpy:latest

# 2. Ejecutar el contenedor
docker run -d -p 5000:5000 --name gui-red-p2p martiniziin/proyectognpy:latest

# 3. Acceder a la aplicación
# Abrir navegador en: http://localhost:5000
```

¡Listo! La aplicación estará disponible en tu navegador.

## 📋 Características

### 🎯 Tres Escenarios de Simulación

#### **Escenario 01: Análisis Básico Punto a Punto**
- Cálculo de presupuesto de potencia óptica
- Análisis de margen de enlace
- Configuración de parámetros de transmisión
- Gráficos interactivos de resultados

#### **Escenario 02: Simulación de Topologías Complejas**
- Carga de archivos de topología JSON
- Visualización interactiva de red
- Análisis con EDFAs y tramos de fibra
- Edición de parámetros en tiempo real
- Cálculos detallados de OSNR

#### **Escenario 03: Visualización en Google Maps**
- Mapeo geográfico de redes ópticas
- Cálculo de rutas óptimas
- Optimización por OSNR o distancia
- Análisis de múltiples rutas alternativas
- Configuración avanzada de parámetros ópticos

### ✨ Tecnologías Integradas

- **[gnpy](https://github.com/Telecominfraproject/oopt-gnpy)**: Motor de simulación óptica profesional
- **[Plotly.js](https://plotly.com/javascript/)**: Visualizaciones interactivas
- **[Google Maps API](https://developers.google.com/maps)**: Mapas geográficos
- **[Bootstrap](https://getbootstrap.com/)**: Interfaz responsiva
- **[Flask](https://flask.palletsprojects.com/)**: Framework web backend

## 🛠️ Gestión del Contenedor

### Comandos Básicos

```bash
# Ver contenedores en ejecución
docker ps

# Ver logs de la aplicación
docker logs gui-red-p2p

# Detener el contenedor
docker stop gui-red-p2p

# Reiniciar el contenedor
docker start gui-red-p2p

# Eliminar el contenedor
docker rm gui-red-p2p
```

### Configuración Avanzada

```bash
# Ejecutar con más recursos
docker run -d -p 5000:5000 --memory=4g --cpus=2 --name gui-red-p2p martiniziin/proyectognpy:latest

# Usar puerto alternativo
docker run -d -p 8080:5000 --name gui-red-p2p martiniziin/proyectognpy:latest
# Acceder en: http://localhost:8080

# Montar directorio para archivos persistentes
docker run -d -p 5000:5000 -v $(pwd)/uploads:/app/uploads --name gui-red-p2p martiniziin/proyectognpy:latest
```

## 📁 Archivos de Topología

### Formatos Soportados
- **Archivos JSON** con definiciones de red OpenROADM
- **Elementos soportados**: Transceivers, EDFAs, Fibras, ROADMs
- **Ejemplos incluidos**: Topologías de Suecia, CORONET, redes mesh

### Estructura de Archivos JSON
```json
{
  "elements": [
    {
      "uid": "transceiver_01",
      "type": "Transceiver",
      "metadata": { "location": { "latitude": 59.3293, "longitude": 18.0686 } }
    },
    {
      "uid": "edfa_01", 
      "type": "Edfa",
      "type_variety": "std_medium_gain"
    }
  ],
  "connections": [
    { "from_node": "transceiver_01", "to_node": "edfa_01" }
  ]
}
```

## 🔧 Guía de Uso Detallada

### Escenario 01: Análisis Básico 
1. Acceder a la pestaña **"Escenario 01"**
2. Configurar parámetros:
   - **Potencia TX (dBm)**: Potencia del transmisor
   - **Sensibilidad RX (dBm)**: Umbral del receptor
   - **Parámetros de Fibra**: Longitud, pérdidas, conectores, número de tramos
3. Hacer clic en **"CALCULAR"**
4. Analizar resultados y gráficos interactivos

### Escenario 02: Simulación de topologías punto a punto
1. Ir a **"Escenario 02"**
2. **Cargar archivo de red**:
   - Seleccionar archivo `.json`
   - Hacer clic en **"Cargar y Visualizar"**
3. **Interactuar con la topología**:
   - Hacer clic en elementos para editar parámetros
   - Ajustar configuraciones de EDFAs
4. **Ejecutar simulación**:
   - Hacer clic en **"CALCULAR"**
   - Revisar tabla de resultados detallados
   - Analizar gráficos de señal, ASE y OSNR

### Escenario 03: Análisis Geográfico
1. Abrir **"Escenario 03"**
2. **Configurar parámetros de ruta**:
   - Seleccionar nodo origen y destino
   - Establecer número de rutas a calcular
   - Elegir criterio de optimización (OSNR/Distancia)
3. **Ajustar parámetros ópticos**:
   - Rango de frecuencias
   - Espaciado de canales
   - Potencia y márgenes del sistema
4. **Calcular y visualizar**:
   - Hacer clic en **"Calcular Rutas"**
   - Revisar tabla comparativa de rutas
   - Explorar visualización en Google Maps

## 🐛 Solución de Problemas

### Problemas Comunes

| Problema | Síntoma | Solución |
|----------|---------|----------|
| **Puerto ocupado** | Error "port already allocated" | Usar puerto diferente: `-p 5001:5000` |
| **Contenedor no inicia** | No responde en localhost:5000 | Verificar logs: `docker logs gui-red-p2p` |
| **Aplicación lenta** | Cálculos tardan mucho | Asignar más recursos: `--memory=4g --cpus=2` |
| **Error de archivos** | No puede cargar topología | Verificar formato JSON válido |
| **Google Maps no carga** | Mapa en blanco en Escenario 03 | Refrescar página, verificar conexión |

### Comandos de Diagnóstico

```bash
# Verificar estado de Docker
docker info

# Inspeccionar contenedor
docker inspect gui-red-p2p

# Acceder al contenedor para debug
docker exec -it gui-red-p2p bash

# Ver uso de recursos
docker stats gui-red-p2p
```

## 📊 Especificaciones Técnicas

### Requisitos del Sistema
- **Sistema Operativo**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: Mínimo 4GB, Recomendado 8GB
- **Almacenamiento**: 2GB libres
- **Red**: Conexión a internet para Google Maps

### Imagen Docker
- **Tamaño**: ~714MB
- **Base**: Python 3.10 slim
- **Arquitectura**: Multi-plataforma (amd64, arm64)
- **Seguridad**: Usuario no-root, puertos mínimos expuestos

### Dependencias Principales
- **Flask 2.x**: Framework web
- **gnpy**: Simulación de redes ópticas
- **numpy**: Cálculos numéricos
- **plotly**: Visualizaciones
- **python-dotenv**: Gestión de configuración

## 📈 Casos de Uso

### 🎓 **Académico**
- Enseñanza de redes ópticas
- Proyectos de investigación
- Simulaciones educativas
- Análisis comparativo de topologías

### 🏢 **Profesional**
- Planificación de redes ópticas
- Análisis de presupuesto de potencia
- Optimización de rutas
- Validación de diseños de red

### 🔬 **Investigación**
- Pruebas de algoritmos de enrutamiento
- Análisis de rendimiento de red
- Comparación de tecnologías ópticas
- Desarrollo de nuevas topologías

## 🤝 Contribuciones

Este proyecto utiliza componentes de código abierto:
- **gnpy**: [Telecom Infra Project](https://github.com/Telecominfraproject/oopt-gnpy)
- **Plotly**: [Plotly Technologies](https://plotly.com/)
- **Bootstrap**: [Bootstrap Team](https://getbootstrap.com/)


## 🆘 Soporte

### Recursos de Ayuda
- **Logs del contenedor**: `docker logs gui-red-p2p`
- **Documentación gnpy**: [gnpy.readthedocs.io](https://gnpy.readthedocs.io/)
- **Issues de GitHub**: [Reportar problemas](../../issues)

### Desinstalación Completa

```bash
# Detener y eliminar contenedor
docker stop gui-red-p2p
docker rm gui-red-p2p

# Eliminar imagen
docker rmi martiniziin/proyectognpy:latest

# Limpiar sistema Docker
docker system prune
```

---

## 🚀 ¡Comienza Ahora!

```bash
docker run -d -p 5000:5000 --name gui-red-p2p martiniziin/proyectognpy:latest
```

**➡️ Abre tu navegador en [http://localhost:5000](http://localhost:5000) y comienza a simular redes ópticas!**

---

<div align="center">

**[⬆️ Volver al inicio](#-gui-red-p2p---simulación-de-redes-ópticas)**

*Desarrollado con ❤️ para la comunidad de redes ópticas*

</div>
