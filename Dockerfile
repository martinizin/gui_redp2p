# Usar la imagen oficial de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de tu proyecto al contenedor
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que Flask escuchará
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
