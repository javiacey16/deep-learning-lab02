# Lab 03 – Redes generativas antagónicas (GANs)

Dos notebooks sobre GANs con MNIST:

| Notebook | Tema | Idioma |
|---|---|---|
| [STUDENT_Dcgan_mnist.ipynb](STUDENT_Dcgan_mnist.ipynb) | Implementación y entrenamiento de una **DCGAN** (64×64) y experimentos con el ruido de entrada | Español |
| [GAN_notebook.ipynb](GAN_notebook.ipynb) | Laboratorio interactivo: GANs en 2D, DCGAN (28×28), exploración del espacio latente, *mode collapse* y GAN condicional | Inglés |

Los dos notebooks están completos y ejecutados, con las respuestas a las preguntas a continuación de cada una. La instalación del entorno se describe en el [README general](../README.md).

## 1. DCGAN en MNIST (`STUDENT_Dcgan_mnist.ipynb`)

**Código completado:**
- Inicialización de pesos: $\mathcal{N}(0, 0.02)$ en las convoluciones, $\mathcal{N}(1, 0.02)$ en BatchNorm y sesgo a 0.
- Generador: 5 convoluciones traspuestas (`nz` → 512 → 256 → 128 → 64 → 1), con BatchNorm y ReLU, y `Tanh` a la salida.
- Discriminador: 5 convoluciones con stride, con BatchNorm y LeakyReLU, y `Sigmoid` a la salida.
- Bucle de entrenamiento (actualización de $D$ con lotes reales y falsos, y de $G$ con la pérdida no saturante), los experimentos de ruido y las comparaciones real/falso.
- Las celdas de Google Colab quedan comentadas para poder ejecutarlo en local.

**Resultados** (5 epochs, lr = 0,005, como indica la práctica):
- Epochs 1-3: el generador produce dígitos realistas y variados (`results/baseline/epoch_1.png` a `epoch_3.png`).
- Último epoch: el entrenamiento diverge. El discriminador gana ($D(x)\to1$, $D(G(z))\to0$), la pérdida de $G$ se dispara y se produce un ***mode collapse* total**: la misma imagen para cualquier ruido (`results/baseline/epoch_4.png`).
- Por eso los experimentos finales con ruido (valores fijos, suma de vectores) se hacen con un generador colapsado. Las respuestas lo explican y lo comparan con lo esperado para un generador bien entrenado.
- El enunciado ya advierte que 0,005 es una tasa de aprendizaje elevada: con 0,001 o 0,0002 el entrenamiento sería más estable.

## 2. Laboratorio interactivo de GANs (`GAN_notebook.ipynb`)

El código venía completo. Se ha ejecutado y se han añadido celdas que reproducen por código los experimentos interactivos (*widgets*) y muestran figuras estáticas, para poder leer los resultados en el notebook guardado, además de las respuestas a todas las preguntas 🔎.

| Parte | Resultado principal |
|---|---|
| 1. GANs en 2D | Baseline: 8/8 modos. Pérdida minimax + SGD: G atascado unas 650 iteraciones. *Mode collapse* con G 10× más rápido que D (0-1/8 modos). WGAN-GP: 8/8 modos |
| 1.3 JS vs Wasserstein | La divergencia JS es constante (log 2) en cuanto las distribuciones no se solapan; $W_1$ crece linealmente con la distancia |
| 2. DCGAN (5 epochs) | Dígitos legibles; $D(x)\approx0{,}7$ y $D(G(z))\approx0{,}3$, sin llegar al equilibrio de 0,5 |
| 3. Espacio latente | Interpolaciones suaves; espacio latente *entangled*; el generador es determinista |
| 4. *Mode collapse* | Los 10 dígitos aparecen (5-14 %), con el 1 infrarrepresentado. El experimento "roto" (G 40× más rápido) genera un 49 % de treses |
| 5. GAN condicional | El clasificador coincide con el dígito pedido en el 89,9 % de las muestras |

## Datos y resultados generados

No se incluyen en el repositorio (ver `.gitignore`); se regeneran al ejecutar los notebooks:
- `mnist/` y `data/`: MNIST, descargado automáticamente por torchvision.
- `gan_lab_outputs/`: modelos entrenados de `GAN_notebook.ipynb`. Si existen, el notebook los carga en lugar de volver a entrenar (`RETRAIN = True` para forzarlo).
- `results/baseline/*.pth`: pesos de la DCGAN. Sí se incluyen las imágenes por epoch (`epoch_*.png`) y las curvas de pérdida (`*.npy`), que el notebook utiliza.

## Tiempos de ejecución aproximados

Con una GPU modesta (2 GB):
- `STUDENT_Dcgan_mnist.ipynb`: unos 40 minutos, porque el código desactiva cuDNN para que los resultados sean reproducibles.
- `GAN_notebook.ipynb`: unos 20 minutos.
