# Deep Learning – Laboratorios

Prácticas de la asignatura de Deep Learning, implementadas en PyTorch. Cada carpeta contiene los notebooks completos y ejecutados (con salidas, figuras y respuestas a las preguntas) y su propio README con los resultados.

| Carpeta | Tema | Notebooks |
|---|---|---|
| [Lab02](Lab02/README.md) | Atención en visión (CBAM) y segmentación semántica (U-Net) | `STUDENT_Lab_Attention_CV.ipynb`, `STUDENT_Lab_Image_Segmentation.ipynb` |
| [Lab03](Lab03/README.md) | Redes generativas antagónicas (GAN, DCGAN, WGAN-GP, cGAN) sobre MNIST | `STUDENT_Dcgan_mnist.ipynb`, `GAN_notebook.ipynb` |

## Resumen de resultados

| Lab | Experimento | Resultado |
|---|---|---|
| Lab02 | CBAM: detectar un parche en imágenes STL-10 | 99,2 % de accuracy en test (85,2 % sin atención) |
| Lab02 | U-Net en Cityscapes (64×64, 10 clases) | 84,6 % de pixel accuracy en test |
| Lab03 | DCGAN en MNIST con lr = 0,005 | Dígitos realistas en los epochs 1-3; *mode collapse* total en el último epoch |
| Lab03 | DCGAN interactiva en MNIST (lr = 2·10⁻⁴) | Genera los 10 dígitos (5-14 % cada uno); cGAN con un 89,9 % de acierto en el dígito pedido |

## Requisitos

- Python 3.10 o superior.
- Recomendado: GPU NVIDIA con CUDA. Todo funciona también en CPU, aunque los entrenamientos son bastante más lentos.
- Espacio en disco: ~5 GB para PyTorch con CUDA y ~6 GB para los datos del Lab02 (STL-10). MNIST (Lab03) ocupa unos 100 MB.

Versiones con las que se ejecutaron los notebooks:

| Librería | Versión |
|---|---|
| torch / torchvision | 2.14.1 / 0.29.1 (CUDA 12.6) |
| scikit-learn | 1.9.1 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| opencv-python | 5.1.0 |

## Instalación

Crear un entorno virtual (en este ejemplo, `.venv` en la raíz del repositorio; puede estar en cualquier otra ruta):

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
.venv/Scripts/python -m pip install jupyter ipykernel ipywidgets matplotlib numpy pandas scikit-learn tqdm pillow opencv-python
```

```bash
.venv/Scripts/python -m ipykernel install --user --name dl_lab02 --display-name "Python (dl_lab02)"
```

Después, en VS Code o Jupyter, selecciona el kernel **"Python (dl_lab02)"**.

## Ejecutar un notebook completo

Desde Jupyter o VS Code, o desde la línea de comandos (el notebook se ejecuta en su propia carpeta). Por ejemplo:

```bash
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=dl_lab02 Lab03/GAN_notebook.ipynb
```

Los datos y los resultados generados (datasets descargados, pesos entrenados) no se incluyen en el repositorio: ver `.gitignore` y el README de cada carpeta.
