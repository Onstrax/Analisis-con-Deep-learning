# Entrega de la persona 1 — Taller 4

## Estado de avance

| Actividad | Estado |
|---|---|
| 1. Exploración y CNN de referencia | **Terminada** |
| 2.1. VAE | **Terminada** |
| 2.2. DDPM | **Terminada** |
| 2.3. Generación de 300 imágenes por modelo | **Terminada** |
| 3. Evaluación de las imágenes generadas | **Pendiente: persona 2** |
| 4. Clasificadores con sobremuestreo y aumento sintético | **Pendiente: persona 2** |
| 5. Análisis y recomendación al laboratorio | **Pendiente: persona 2** |
| Bono GAN | **Pendiente: persona 2** |

La ejecución registrada se hizo en CUDA con una NVIDIA GeForce RTX 3060 Ti, Python 3.13.2 y PyTorch 2.14.1+cu130. Se usaron MedMNIST 3.0.2 y diffusers 0.43.0. Los valores y resultados de esta entrega corresponden a los archivos de configuración, checkpoints y salidas conservados aquí; no se ejecutó ni reentrenó ningún notebook para redactar este informe.

## 1. Datos y clasificador de referencia

Se usó BloodMNIST en sus particiones oficiales. Las imágenes son RGB de 28×28, originalmente en el rango `[0,255]`. El conteo de basófilos en entrenamiento es **N = 852**. Con `np.random.default_rng(42)` se seleccionaron sin reemplazo 200 índices de basófilos dentro de `train`; esos 200 se usaron para entrenar el clasificador y ambos generadores. La CNN de referencia usó además todas las imágenes de entrenamiento de las otras siete clases (11.107 imágenes).

| Clase | Entrenamiento | Validación | Prueba |
|---|---:|---:|---:|
| Basófilo | 852 | 122 | 244 |
| Eosinófilo | 2.181 | 312 | 624 |
| Eritroblasto | 1.085 | 155 | 311 |
| Granulocito inmaduro | 2.026 | 290 | 579 |
| Linfocito | 849 | 122 | 243 |
| Monocito | 993 | 143 | 284 |
| Neutrófilo | 2.330 | 333 | 666 |
| Plaqueta | 1.643 | 235 | 470 |

La inspección de las imágenes pequeñas sugirió un núcleo central oscuro y violáceo, con poco detalle citoplasmático. Los basófilos pueden confundirse visualmente con linfocitos y algunos granulocitos inmaduros; seis imágenes por clase y la resolución limitan esta observación.

La CNN tiene dos bloques `Conv2d(3,32,3)-ReLU-MaxPool2d(2)` y `Conv2d(32,64,3)-ReLU-MaxPool2d(2)`, seguidos de `Flatten`, `Linear(1600,128)-ReLU-Dropout(0.3)` y una capa final de ocho logits. Se entrenó con Adam (`lr=0.001`), lotes de 128, hasta 10 épocas y `EarlyStopping` sobre `val_loss` con paciencia 3 y restauración de los mejores pesos. La entrada RGB se convirtió de `[0,255]` a `[0,1]`. Se completaron las 10 épocas; la mejor `val_loss` fue 0.472257.

Resultados en la partición oficial de prueba:

| Métrica | Resultado |
|---|---:|
| F1 macro | 0.769993 |
| F1 de basófilo | 0.331081 |
| Recall de basófilo | 0.200820 |

El checkpoint de referencia está en `artefactos/persona1/cnn_referencia.pt`. Es un checkpoint PyTorch con `state_dict`, historial, número de épocas y configuración. Las métricas, predicciones (`y_true`, `y_pred`) e historial también están en `artefactos/persona1/metricas_cnn.json`. La clase y el cargador definidos en el notebook son `CNNReferencia` y `cargar_cnn_referencia(...)`; este último carga los pesos con `weights_only=True`. Para reutilizar sus predicciones sobre datos sintéticos, convertirlos a RGB `[0,255]` y luego aplicar exactamente la conversión de entrada de la CNN:

```python
modelo_cnn, checkpoint_cnn = cargar_cnn_referencia()
x_sintetico = imagenes_a_tensor(imagenes_rgb_uint8, "cero_uno").to(DISPOSITIVO)
with torch.no_grad():
    predicciones = modelo_cnn(x_sintetico).argmax(dim=1).cpu().numpy()
```

`idx_sub.npy` contiene posiciones relativas a `train`, no imágenes ni etiquetas. Para restaurar la selección fija, calcule `N` desde las etiquetas oficiales y use esos índices, sin volver a sortear:

```python
idx_sub = np.load(ARTEFACTOS / "idx_sub.npy")
labels_train = train.labels.ravel()
N = int((labels_train == 0).sum())  # 852
idx_otros = np.flatnonzero(labels_train != 0)
assert len(idx_sub) == 200 and np.all(labels_train[idx_sub] == 0)
basofilos_train = train.imgs[idx_sub]
```

Para la actividad 4, el conjunto con sobremuestreo debe conservar todas las imágenes de las otras siete clases y completar N basófilos repitiendo los 200 reales:

```python
idx_baso_sobremuestreo = np.resize(idx_sub, N)
idx_train_sobremuestreo = np.concatenate((idx_baso_sobremuestreo, idx_otros))
```

En la variante con aumento sintético se combinan los 200 basófilos reales, las mismas imágenes de `idx_otros` y **N − 200 = 652** imágenes sintéticas del generador elegido en la actividad 3. Mantenga validación y prueba en sus particiones oficiales: no entrene con `val` ni con `test`, y evalúe los tres clasificadores sobre el mismo `test`.

## 2. Generadores y resultados

Ambos generadores se entrenaron exclusivamente con los mismos 200 basófilos reales seleccionados de `train`. La conversión de entrada es RGB `[0,255]` a tensores flotantes NCHW en `[-1,1]`. La salida se limita a ese rango y se convierte nuevamente a PNG RGB de 28×28 en `[0,255]`.

### 2.1 VAE

- **Arquitectura:** espacio latente de dimensión 16. El encoder usa `Conv2d(3,32,3,stride=2,padding=1)-ReLU` y `Conv2d(32,64,3,stride=2,padding=1)-ReLU` para reducir 28→14→7. Capas lineales desde 3.136 características producen `mu` y `logvar` de dimensión 16. El decoder proyecta de 16 a 3.136, reconstruye 7→14→28 mediante `ConvTranspose2d(64,32,4,stride=2,padding=1)` y `ConvTranspose2d(32,16,4,stride=2,padding=1)`, y termina con `Conv2d(16,3,3,padding=1)-tanh`.
- **Entrenamiento:** Adam, tasa `1e-3`, lote 32, semilla 42, máximo 300 épocas o 1.200 segundos de GPU. Se completaron 300 épocas en **13.072 s de GPU**.
- **Pérdida:** por imagen, suma de los errores cuadráticos de reconstrucción (MSE) más la divergencia KL sumada en la dimensión latente; luego se promedia entre los elementos del lote. Última pérdida total **49.780727**, reconstrucción **28.089805** y KL **21.690921**.
- **Checkpoint:** `artefactos/persona1/vae.pt`, con `state_dict`, configuración, historial, tiempo y épocas. El notebook proporciona `cargar_vae(...)`, que reconstruye la arquitectura indicada por `latent_dim` y carga sus pesos.
- **Imágenes y muestreo:** 300 PNG en `artefactos/persona1/vae_imagenes/000.png` a `299.png`. El muestreo de esas 300 tardó **0.0234 s**. `generar_vae(modelo, cantidad, batch_size=64, seed=123, device=DISPOSITIVO)` permite pedir otra cantidad; devuelve un tensor NCHW en `[-1,1]`.

### 2.2 DDPM

- **Arquitectura:** `UNet2DModel` para 28×28, con 3 canales de entrada y salida, un nivel por bloque, `block_out_channels=[64,128]`, `DownBlock2D` y `AttnDownBlock2D`, `AttnUpBlock2D` y `UpBlock2D`, y `norm_num_groups=32`.
- **Difusión y entrenamiento:** `DDPMScheduler` con 1.000 pasos de entrenamiento; la U-Net predice el ruido añadido y se optimiza el MSE entre ese ruido y la predicción. AdamW, tasa `1e-4`, lote 32, semilla 42, máximo 300 épocas y tope de 3.600 segundos de GPU. Se completaron **300 épocas y 2.100 actualizaciones reales de `optimizer.step()`** en **112.037 s de GPU**. Los 1.000 pasos del scheduler son pasos del proceso de difusión, no actualizaciones de parámetros. La pérdida MSE final fue **0.030810**.
- **Checkpoint:** directorio `artefactos/persona1/ddpm/`, con configuración de diffusers, `diffusion_pytorch_model.safetensors`, subdirectorio `scheduler/` y `training_metadata.pt` (configuración, historial, tiempo, épocas, actualizaciones y estado parcial). `cargar_ddpm(...)` carga la U-Net con `UNet2DModel.from_pretrained`, el scheduler con `DDPMScheduler.from_pretrained` y los metadatos con `torch.load(..., weights_only=True)`.
- **Imágenes y muestreo:** 300 PNG en `artefactos/persona1/ddpm_imagenes/000.png` a `299.png`; se usaron **50 pasos de inferencia** para generarlas. El muestreo de esas 300 tardó **12.487 s**. `generar_ddpm(modelo, cantidad, pasos_inferencia=50, batch_size=16, seed=456, device=DISPOSITIVO, scheduler=...)` permite generar una cantidad arbitraria y devuelve NCHW en `[-1,1]`. Los 50 pasos de inferencia no son los 1.000 pasos de entrenamiento.

El código real de carga y generación está en el notebook `Taller 4/juan_alvarez_daniel_ortiz.ipynb`. Para obtener 652 imágenes con el modelo seleccionado por la persona 2 en su evaluación:

```python
modelo_vae, datos_vae = cargar_vae()
vae_nuevas = generar_vae(modelo_vae, N - 200)

modelo_ddpm, scheduler_ddpm, metadata_ddpm = cargar_ddpm()
ddpm_nuevas = generar_ddpm(
    modelo_ddpm, N - 200, pasos_inferencia=50, scheduler=scheduler_ddpm
)
```

Genere con el modelo elegido, no es necesario entrenarlo de nuevo. La función devuelve imágenes en `[-1,1]`; antes de pasarlas a la CNN o combinarlas con imágenes RGB, conviértalas al formato esperado:

```python
imagenes_rgb_uint8 = (
    (tensor_sintetico.clamp(-1, 1) + 1) * 127.5
).round().byte().permute(0, 2, 3, 1).cpu().numpy()
x_sintetico_cnn = imagenes_a_tensor(imagenes_rgb_uint8, "cero_uno")
```

El notebook conserva las funciones para guardar PNG y para cargar checkpoints; las celdas que encuentran los checkpoints guardados los cargan en vez de volver a entrenar. Si se piden 652 imágenes directamente al generador, son 652 muestras nuevas; para reutilizar las 300 PNG ya conservadas y completar el conjunto, cargue esas muestras y genere solo las 352 adicionales.

### Observación visual y límites de lo completado

En las cuadrículas guardadas, ambos modelos muestran regiones centrales violáceas con aspecto celular, pero con bordes y detalles finos suaves y poco definidos a 28×28. En las 16 muestras inspeccionadas, el VAE se ve más suavizado; DDPM presenta algo más de variación de forma y textura, aunque también mantiene contornos suaves. Es una apreciación descriptiva, no una evaluación formal de realismo o diversidad. Los gráficos están en `artefactos/persona1/graficas/`, entre ellos `muestras_reales_48.png`, `muestras_vae.png` y `muestras_ddpm.png`.

No se han calculado todavía las medidas de la actividad 3: porcentaje de sintéticas clasificadas como basófilos, distancias entre pares, vecinos más cercanos ni análisis de posible memorización. Tampoco se seleccionó un generador para el aumento, se entrenaron las dos CNN comparativas de la actividad 4, se elaboró el análisis/recomendación de la actividad 5 ni se entrenó una GAN. En consecuencia, no hay evidencia aquí de que el aumento sintético mejore la clasificación, de que las imágenes sean privadas o de que un modelo sea preferible. Los resultados disponibles son de una sola semilla, dependen de solo 200 ejemplos para generar una clase poco frecuente y describen imágenes de muy baja resolución. La CNN evaluadora comparte esos 200 ejemplos de entrenamiento con los generadores; su evaluación de consistencia no es independiente. Las métricas de píxeles tampoco bastan por sí solas para juzgar plausibilidad biológica o privacidad. BloodMNIST y estos experimentos son académicos y no sustentan decisiones clínicas.

## Continuación de las actividades por la persona 2

1. Use `idx_sub.npy` para reconstruir los mismos 200 índices respecto a `train`; no los vuelva a sortear. Calcule **N = 852** desde `train.labels`. Para la actividad 3, inspeccione 16 imágenes de cada generador y 16 basófilos de `test`, clasifique las 300 sintéticas con la CNN de referencia y compare la proporción clasificada como basófilo con el recall de esa CNN en los basófilos de prueba. Calcule además las distancias solicitadas sobre 100 imágenes sintéticas y 100 basófilos reales de prueba, y compare los vecinos más cercanos de las sintéticas con los basófilos de prueba respecto a los mismos 200 ejemplos de entrenamiento.
2. Justifique la elección de generador con esa evaluación antes de crear el conjunto de aumento sintético. Se necesitan **652 = N − 200** sintéticas para que haya N basófilos. Los 200 reales y todas las imágenes de las otras clases deben ser las mismas de la actividad 1. Para el conjunto de sobremuestreo, repita los 200 basófilos reales hasta N; para el de aumento, use 200 reales más las 652 sintéticas. Mantenga intactas las particiones oficiales de validación y prueba, el formato RGB y el preprocesamiento de la CNN. Entrene ambos clasificadores con la misma arquitectura, configuración y semilla de la referencia, y evalúe los tres en el mismo `test`.
3. La actividad 5 debe interpretar los resultados, la variabilidad de una sola semilla y las limitaciones de las métricas y del experimento; no extrapole a uso clínico ni a privacidad. El bono GAN no está implementado; el enunciado propone Adam, `lr=2e-4`, `betas=(0.5, 0.999)` y máximo 20 minutos de GPU, y solicita comparación equivalente con los otros generadores.

## Entorno y ejecución en VS Code

La ejecución original registrada usó Python 3.13.2, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, diffusers 0.43.0 y MedMNIST 3.0.2. En ese entorno también estaban numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1, scikit-learn 1.9.0, Pillow 12.3.0, ipykernel 7.3.0, nbformat 5.11.1 y nbconvert 7.17.2. Para recrearlo en Windows, instale y seleccione Python 3.13.2 y ejecute en PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version  # Debe indicar Python 3.13.2
python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install diffusers==0.43.0 medmnist==3.0.2 numpy==2.5.2 pandas==3.0.5 matplotlib==3.11.1 scikit-learn==1.9.0 Pillow==12.3.0 ipykernel==7.3.0 nbformat==5.11.1 nbconvert==7.17.2
```

Si `py -3.13` no selecciona Python 3.13.2, seleccione la instalación exacta de 3.13.2 antes de crear el entorno. La instalación de PyTorch CUDA usa el índice oficial para CUDA 13.0; la disponibilidad efectiva de CUDA también depende de la GPU y del controlador.

En VS Code, abra el repositorio y `Taller 4/juan_alvarez_daniel_ortiz.ipynb`, seleccione como intérprete y kernel `.venv\Scripts\python.exe`, y confirme que el directorio de trabajo es `Taller 4` (el notebook usa `Path.cwd()` como raíz de los artefactos). La extensión de Python y la de Jupyter deben estar habilitadas. Ejecute las celdas en orden para restablecer las definiciones y cargar los resultados/checkpoints guardados; antes de una ejecución completa, confirme que existen los archivos enumerados arriba. No ejecute las celdas de entrenamiento si desea limitarse a completar las actividades pendientes.

## Licencias y referencias

BloodMNIST/MedMNIST indica **CC BY 4.0 para el conjunto de datos**. La licencia **Apache-2.0 del repositorio MedMNIST corresponde a su código y no sustituye la licencia de los datos**. Al reutilizar los datos, atribuya la fuente y conserve la referencia/licencia aplicables:

- Yang, J. et al. (2023). “MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification”. *Scientific Data*, 10, 41. https://doi.org/10.1038/s41597-022-01721-8
- Yang, J., Shi, R. & Ni, B. (2021). “MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis”. *ISBI*. https://doi.org/10.1109/ISBI48211.2021.9434062
- Dataset oficial: https://doi.org/10.5281/zenodo.10519652 ; MedMNIST: https://medmnist.com/ ; repositorio y licencia del código: https://github.com/MedMNIST/MedMNIST
- El DDPM usa Hugging Face diffusers y `UNet2DModel`/`DDPMScheduler`: https://huggingface.co/docs/diffusers/ . Referencia del proceso: Ho, J., Jain, A. & Abbeel, P. (2020). “Denoising Diffusion Probabilistic Models”. *NeurIPS*. https://arxiv.org/abs/2006.11239
