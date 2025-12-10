import cv2
from ultralytics import YOLO
import os

MODEL_NAME = 'yolov8n.pt'
IMAGE_URL = 'https://ultralytics.com/images/zidane.jpg'

def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print("Analizando imagen...")

 # INFERENCIA CON YOLO
    # source: URL o path local de la imagen.
    # save=True: guarda la imagen detectada con cajas dibujadas.
    # conf=0.5: confianza mínima del 50%.
    # project/name: define dónde se guardará el resultado.    
    results = model.predict(
        source=IMAGE_URL,
        save=True,
        conf=0.5,
        project='../01-ComputerVision/output',
        name='resultado_retail'
    )


# YOLO devuelve una lista de resultados.
# Como solo analizamos una imagen, tomamos el primer elemento.
    result = results[0]

    print("\n--- REPORTE DE DETECCIÓN ---")
# IDs de clases detectadas por YOLO (ej: 0=persona, 2=carro…)
    box_cls = result.boxes.cls.tolist()
# Diccionario con nombres de clases (COCO: 80 clases)
    nombres = result.names


 # CONTAR OBJETOS DETECTADOS
    conteo = {}
    for clase_id in box_cls:
        nombre = nombres[int(clase_id)]
        conteo[nombre] = conteo.get(nombre, 0) + 1
# Mostrar el conteo en consola
    for obj, cant in conteo.items():
        print(f"-> Se detectaron {cant} {obj}(s)")

    # --- OBTENER RUTA REAL DE LA IMAGEN GUARDADA ---
    final_path = result.save_dir + "/" + os.path.basename(result.path)

    print(f"\nImagen guardada en: {final_path}")

    img = cv2.imread(final_path)

# Validación por si algo falló al guardar/cargar la imagen
    if img is None:
        print("⚠ ERROR: No se pudo leer la imagen guardada.")
        print("Contenido del directorio:")
        print(os.listdir(result.save_dir))
        return


 # MOSTRAR IMAGEN CON OPENCV
    cv2.imshow("Detector Retail - YOLOv8", img)
    print("Presiona cualquier tecla para salir.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Punto de entrada del script
if __name__ == "__main__":
    main()
