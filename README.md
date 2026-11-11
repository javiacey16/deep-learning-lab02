# Deep Learning – Lab 02

Dos prácticas de visión por computador con PyTorch:

| Notebook | Tema | Resultado (test) |
|---|---|---|
| [STUDENT_Lab_Attention_CV.ipynb](STUDENT_Lab_Attention_CV.ipynb) | **Atención en visión (CBAM)**: clasificar si una imagen de STL-10 tiene un parche añadido | Accuracy **99,2 %** con CBAM frente a 85,2 % sin atención |
| [STUDENT_Lab_Image_Segmentation.ipynb](STUDENT_Lab_Image_Segmentation.ipynb) | **Segmentación semántica con U-Net** sobre Cityscapes (64×64, 10 clases obtenidas con K-Means) | Pixel accuracy **84,6 %** |

Los dos notebooks están completos y ejecutados, con salidas, figuras y la discusión de resultados (en inglés, como el enunciado).

## Contenido

### 1. Attention in Computer Vision (CBAM)
- Implementación del **Convolutional Block Attention Module**: Channel Attention Module (CAM) seguido del Spatial Attention Module (SAM).
- Red convolucional sencilla + CBAM, entrenada 30 épocas con `BCELoss` y Adam.
- Visualización de los mapas de características refinados y de la máscara de atención espacial, que localiza el parche.
- **Ejercicio opcional:** comparación con la misma red sin atención.

| Partición | CBAM | Sin atención |
|---|---|---|
| Train | 0,993 | 0,846 |
| Validación | 0,998 | 0,861 |
| Test | 0,992 | 0,852 |

### 2. Image Segmentation (U-Net)
- Separación de imagen y etiqueta en el dataset Cityscapes (pix2pix), y conversión de colores RGB a 10 clases con **K-Means**.
- U-Net completa (encoder, bottleneck, decoder con skip connections y clasificador) con BatchNorm, unos 31 M de parámetros.
- Entrenamiento de 30 épocas con `CrossEntropyLoss` y Adam, y visualización de las predicciones.

| Partición | Pixel accuracy |
|---|---|
| Train | 0,903 |
| Validación | 0,896 |
| Test | 0,846 |

## Requisitos

- Python 3.10 o superior.
- Recomendado: GPU NVIDIA con CUDA. Funciona también en CPU, pero el entrenamiento de la U-Net es mucho más lento.
- Unos 5 GB libres para PyTorch con CUDA y unos 6 GB para los datos (STL-10 comprimido y extraído).

Versiones con las que se ejecutaron los notebooks:

| Librería | Versión |
|---|---|
| torch / torchvision | 2.14.1 / 0.29.1 (CUDA 12.6) |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |

## Instalación

Crear un entorno virtual (en este ejemplo, `.venv` dentro del proyecto; puede estar en cualquier otra ruta):

```bash
python -m venv .venv
```

Instalar PyTorch. Elige el índice adecuado para tu sistema en [pytorch.org](https://pytorch.org/get-started/locally/); por ejemplo, con CUDA 12.6:

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> En Linux/macOS, usa `.venv/bin/python` en lugar de `.venv/Scripts/python`.

> Para GPUs antiguas (arquitectura Pascal, `sm_61`), usa las wheels `cu126`: las versiones `cu128` y posteriores ya no las soportan.

Instalar el resto de dependencias y registrar el kernel de Jupyter:

```bash
.venv/Scripts/python -m pip install jupyter ipykernel matplotlib numpy pandas scikit-learn tqdm ipywidgets pillow
```

```bash
.venv/Scripts/python -m ipykernel install --user --name dl_lab02 --display-name "Python (dl_lab02)"
```

Después, en VS Code o Jupyter, selecciona el kernel **"Python (dl_lab02)"**.

## Datos

| Dataset | Cómo obtenerlo |
|---|---|
| STL-10 | torchvision lo descarga automáticamente (~2,6 GB). Si el servidor de Stanford no responde, existe una copia de `stl10_binary.tar.gz` en Hugging Face (`xingslong/stl10_binary`): colócala en la carpeta de datos y torchvision comprobará su MD5. |
| Cityscapes | Extraer `Cityscapes.zip` (incluido en el repositorio): 2975 imágenes de entrenamiento y 500 de validación. |

Antes de ejecutar, ajusta las rutas a tu carpeta de datos:
- `DATA_ROOT` en el notebook de atención (carpeta donde se descarga STL-10).
- `path_to_folder` en el notebook de segmentación (carpeta `Cityscapes/cityscapes_data/`).

## Ejecutar los notebooks

Desde Jupyter o VS Code, o completos desde la línea de comandos:

```bash
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=dl_lab02 STUDENT_Lab_Attention_CV.ipynb
```

```bash
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=dl_lab02 STUDENT_Lab_Image_Segmentation.ipynb
```

Los resultados pueden variar ligeramente entre ejecuciones: los parches de STL-10 y las particiones se generan aleatoriamente, y el entrenamiento en GPU no es determinista.

## Notas sobre el código proporcionado

- **Parche "blanco" de STL-10:** `img[...] = 1` se aplica a una imagen `uint8` (0-255), así que el parche es en realidad casi negro.
- **Partición train/validación:** se crean con dos `random_split` independientes del mismo dataset, de modo que validación y entrenamiento se solapan. La métrica de validación es, por tanto, optimista; el conjunto de test es la estimación fiable.
- **Modo evaluación:** se añadió `model.eval()` en validación y evaluación para que BatchNorm no actualice sus estadísticas con esos datos.
