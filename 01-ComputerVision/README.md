Este repositorio contiene en caso de uso de Vision por cmputadoras usando YOLO.

Pasos Iniciales:

# Crear entorno virtual (Windows)
python -m venv venv
.\venv\Scripts\activate

# Instalar librerías
pip install ultralytics opencv-python








## Descripción
Este proyecto implementa un sistema de visión artificial capaz de detectar, clasificar y contar objetos en tiempo real utilizando la arquitectura **YOLOv8** (You Only Look Once).

El objetivo es simular un escenario de análisis en tienda o seguridad industrial, donde se requiere identificar personas y objetos automáticamente.

## Tecnologías Utilizadas
* **Python 3.10+**
* **YOLOv8 (Ultralytics):** Modelo SOTA (State of the Art) para detección de objetos.
* **OpenCV:** Para pre-procesamiento y visualización de imágenes.
* **Dataset:** COCO (Common Objects in Context) pre-entrenado.

## Estructura
* `src/detect.py`: Script principal de inferencia y conteo.
* `output/`: Carpeta generada automáticamente con las imágenes analizadas.

## Cómo ejecutar
1. Instalar dependencias:
   ```bash
   pip install ultralytics opencv-python







# Ejecutar
python src/detect.py