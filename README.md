# Proyecto de Fine-Tuning de Modelos 3D de Gafas

Este proyecto realiza un ajuste fino (fine-tuning) de un modelo preentrenado para generar modelos 3D de gafas a partir de imágenes.

## Estructura del proyecto

- `data/`: Contiene los datos de entrenamiento y prueba.
- `model/`: Contiene el modelo preentrenado.
- `checkpoints/`: Almacena los modelos ajustados.
- `scripts/`: Contiene el script para el ajuste fino.

## Cómo ejecutar el proyecto

1. Asegúrate de tener las dependencias necesarias: `pip install torch torchvision`.
2. Organiza tus imágenes en las carpetas `data/train` y `data/test`.
3. Carga tu modelo preentrenado en `model/`.
4. Ejecuta el script: `python scripts/fine_tuning.py`.
