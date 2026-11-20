# Plan de implementación y handoff — Taller 4

Este documento audita el estado actual del Taller 4 y define un plan autocontenido para que una nueva sesión, sin acceso al historial de esta conversación, pueda completar las actividades 3, 4, 5 y el bono GAN. **Esta sesión no implementa código ni modifica entregables del taller**: solo diagnostica, planifica y escribe este archivo.

---

## 0. Estado final de la sesión posterior

Las secciones siguientes conservan el diagnóstico y el plan originales como registro histórico. La ejecución final ya está completada:

- Se recuperó `MINE_4210_ADL_202620_L10.ipynb`. De él se adoptaron para la GAN la inicialización DCGAN (`Conv/ConvTranspose N(0,0.02)`, `BatchNorm N(1,0.02)`), etiquetas reales `0.9`, `BCEWithLogitsLoss`, Adam `lr=2e-4`, `betas=(0.5,0.999)` y snapshots con ruido fijo (`graficas/evolucion_gan.png`, `modelos/gan/snapshots_gan.pt`). No se adoptaron FD/DDIM por no estar en la rúbrica y por la propia advertencia del Lab10 sobre memorización.
- La GAN se reentrenó en 61,634 s de GPU (600 épocas, 2400 actualizaciones, límite 1200 s cumplido), sin duplicados ni indicios de colapso; diversidad 7.863 frente a 8.194 real y vecino train 5.827 frente a 5.766 test.
- La decisión final cambió de DDPM a **GAN** (puntaje ponderado mayor). El aumento sintético usó 300 GAN + 352 GAN adicionales (semilla 1303), y las CNN de sobremuestreo/aumento se reentrenaron con configuración coherente.
- Resultados del test: referencia `0.769993/0.331081/0.200820`, sobremuestreo `0.823533/0.680761/0.659836`, aumento GAN `0.823654/0.672606/0.618852` (F1 macro / F1 basófilo / recall basófilo).
- Nota al laboratorio: 248 palabras (≤300), generada desde variables del notebook. `ENTREGA_PERSONA2.md` es consistente con GAN.
- Notebook `juan_alvarez_daniel_ortiz.ipynb`: 43 celdas, 18 de código, ejecutadas arriba abajo sin errores. `artefactos/persona1/` no se modificó. No se hicieron commits.

---

## 1. Diagnóstico del estado actual

### 1.1 Estado del repositorio

- Raíz del workspace: `/home/juanes/Tareas/9 Semestre/Analisis Deep learning/Talleres`.
- Proyecto del taller: `Taller 4/`.
- Rama actual: `main`, sincronizada con `origin/main`.
- `git status --short --branch` al auditar: limpio, sin cambios pendientes.
- Historial reciente relevante: `7070d68 Taller 4`, `885720e Tessl Etapa 4`, `91a2f36 tee toca`.
- No existe aún `Taller 4/PLAN_IMPLEMENTACION.md` antes de esta escritura, ni carpeta `artefactos/persona2/`.

### 1.2 Inventario verificado

| Ruta | Estado verificado |
|---|---|
| `Taller 4/juan_alvarez_daniel_ortiz.ipynb` | Existe. 21 celdas: 16 markdown y 5 code. Las 5 celdas de código tienen `execution_count` y `outputs`, es decir, están ejecutadas. Conserva el enunciado original, la rúbrica y la solución de la persona 1 hasta la actividad 2.3. |
| `Taller 4/artefactos/persona1/ENTREGA_PERSONA1.md` | Existe y es la fuente autorizada del estado de avance. Declara terminadas las actividades 1, 2.1, 2.2 y 2.3; pendientes 3, 4, 5 y bono para la persona 2. |
| `artefactos/persona1/idx_sub.npy` | Existe. Contiene 200 índices relativos a `train`, todos con etiqueta 0. No contiene imágenes ni etiquetas. |
| `artefactos/persona1/cnn_referencia.pt` | Existe. Checkpoint PyTorch cargable con `weights_only=True`; claves: `state_dict` (8 tensores), `config`, `epocas`, `historial`. |
| `artefactos/persona1/vae.pt` | Existe. Checkpoint cargable; claves: `state_dict` (16 tensores), `config`, `epocas`, `historial`, `tiempo_entrenamiento_segundos`. |
| `artefactos/persona1/ddpm/` | Existe con `config.json`, `diffusion_pytorch_model.safetensors`, `scheduler/scheduler_config.json`, `training_metadata.pt`. Metadatos cargables. |
| `artefactos/persona1/vae_imagenes/` | 300 PNG RGB, `000.png` a `299.png`, 28×28. |
| `artefactos/persona1/ddpm_imagenes/` | 300 PNG RGB, `000.png` a `299.png`, 28×28. |
| `artefactos/persona1/graficas/` | 7 PNG: distribución de clases, muestras reales, pérdidas CNN/VAE/DDPM, muestras VAE/DDPM. |
| `artefactos/persona1/configuracion.json` | Existe. Registra semillas, N, arquitecturas, entrenamientos, tiempos y rutas relativas con separadores de Windows en algunos campos. |
| `artefactos/persona1/metricas_cnn.json` | Existe. Incluye métricas, `y_true`, `y_pred` e historial de la CNN de referencia. |
| Tamaño total de `artefactos/persona1/` | Aprox. 14.7 MB. |

### 1.3 Hechos numéricos que deben respetarse

- Dataset: BloodMNIST, RGB 28×28, 8 clases, particiones oficiales MedMNIST.
- `N = 852` basófilos en `train`; se seleccionaron 200 con `np.random.default_rng(42)` sin reemplazo.
- Otras clases en `train`: 11.107 imágenes.
- Particiones: `train` 11.959, `val` 1.712, `test` 3.421 imágenes.
- Basófilos en `test`: 244.
- CNN de referencia: 10 épocas, mejor `val_loss = 0.472257248709135`.
- Métricas de la CNN en `test`: F1 macro `0.7699926171640274`, F1 basófilo `0.3310810810810811`, recall basófilo `0.20081967213114754`.
- VAE: `latent_dim=16`, Adam `lr=1e-3`, lote 32, 300 épocas, 13.072 s de GPU, pérdida final total `49.780727`, reconstrucción `28.089805`, KL `21.690921`. Muestreo de 300: 0.0234 s.
- DDPM: `UNet2DModel` 28×28, `DDPMScheduler` con 1.000 pasos de entrenamiento, AdamW `lr=1e-4`, lote 32, 300 épocas, 2.100 actualizaciones reales, 112.037 s de GPU, MSE final `0.030810434504279068`. Inferencia: 50 pasos. Muestreo de 300: 12.487 s.
- Los generadores se entrenaron solo con los mismos 200 basófilos de `train`, con RGB `[0,255] -> [-1,1]`.
- La CNN espera RGB `[0,255] -> [0,1]` y devuelve 8 logits; `argmax` da la clase.

### 1.4 Cobertura de requisitos

| Requisito/rúbrica | Estado | Evidencia |
|---|---|---|
| 1. Exploración y CNN de referencia (15 %) | **Completo** | Notebook celdas 9 y 12; `ENTREGA_PERSONA1.md`; `configuracion.json`; `metricas_cnn.json`; checkpoint. |
| 2. VAE + DDPM + 300 imágenes por modelo (30 %) | **Completo** | Notebook celdas 14, 16 y 18; checkpoints; 600 PNG; métricas y tiempos documentados. |
| 3. Evaluación de imágenes generadas (20 %) | **Pendiente** | No hay métricas de consistencia, diversidad, memorización ni selección de generador. |
| 4. CNN con sobremuestreo y aumento sintético (15 %) | **Pendiente** | No hay conjuntos construidos, checkpoints ni tabla comparativa de tres clasificadores. |
| 5. Análisis y nota al laboratorio (20 %) | **Pendiente** | No hay análisis comparativo ni recomendación ≤300 palabras. |
| Bono GAN (+15 %) | **Pendiente** | No hay GAN, imágenes, evaluación equivalente ni análisis de colapso de modos. |
| Entregable: notebook con todas las celdas ejecutadas, organizado por actividades | **Parcial** | El notebook actual ejecuta y documenta solo hasta 2.3. Faltan secciones de persona 2 y del bono. |

### 1.5 Vacíos, ambigüedades y riesgos detectados

1. **Entorno local distinto del documentado.** `ENTREGA_PERSONA1.md` registra Python 3.13.2, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, diffusers 0.43.0, MedMNIST 3.0.2, GPU RTX 3060 Ti. En este checkout Linux, el intérprete del sistema es Python 3.14.10 sin torch/diffusers/medmnist; `.venv/bin/python` es Python 3.14.10 con torch 2.15.0+cu132 pero sin torchvision, diffusers ni medmnist; `venv/bin/python` es Python 3.11.16 sin dependencias clave. Hay GPU NVIDIA RTX 2070 de 8 GB. Antes de ejecutar hay que reconstruir un entorno compatible o instalar las dependencias faltantes. Los tiempos y quizá algunos resultados pueden diferir.
2. **Directorio de trabajo.** El notebook usa `RAIZ = Path.cwd()`. Debe ejecutarse con directorio de trabajo `Taller 4`; si se abre desde otro lugar, escribirá artefactos en la ruta equivocada.
3. **No reentrenar persona 1.** Las celdas ya detectan checkpoints y los cargan, pero la celda 9 vuelve a guardar `idx_sub.npy` de forma determinista. Es inocuo, pero debe evitarse sobrescribir artefactos de persona 1 salvo verificación.
4. **Muestreo no especificado.** El enunciado dice “100 imágenes sintéticas” y “100 basófilos reales de test” sin fijar índices. Decisión recomendada: primeras 100 PNG (`000..099`) y primeros 100 basófilos de `test` en orden oficial, documentando la regla.
5. **“Distancia euclidiana promedio entre pares” es ambigua.** Puede significar dentro de sintéticas, dentro de reales o entre conjuntos. El plan exige calcular y reportar las tres, señalando cuál se usa como diversidad principal.
6. **CNN evaluadora no independiente.** Fue entrenada con los mismos 200 basófilos usados por los generadores. El porcentaje de sintéticas clasificadas como basófilo no prueba realismo ni privacidad.
7. **Código del laboratorio para la GAN no está en el repo.** El enunciado dice “A partir del código del laboratorio” (Laboratorio 10), pero no hay notebook de ese laboratorio en este workspace. Hay que recuperarlo si se quiere adherencia literal; si no, usar una DCGAN compacta documentada en el plan.
8. **Presupuesto de la GAN.** Máximo 20 minutos de GPU. Debe medirse con `torch.cuda.synchronize()` y guardarse checkpoint/metadatos periódicos para no perder el trabajo si se corta.
9. **Una sola semilla.** No permite afirmar que las diferencias entre clasificadores sean estadísticamente confiables; debe discutirse en el análisis.
10. **Filtrado de datos.** Las 652 sintéticas para aumento deben venir solo del generador entrenado con los 200 basófilos de `train`; `val` y `test` deben permanecer intactos.
11. **Reproducibilidad de PNG.** Generar con la misma semilla en otra GPU/versión puede no producir bytes idénticos. Verificar forma, rango y determinismo funcional, no igualdad byte a byte.
12. **Licencias.** BloodMNIST es CC BY 4.0; el código MedMNIST es Apache-2.0. La entrega debe conservar citas y la advertencia de uso académico, sin conclusiones clínicas.

---

## 2. Plan de implementación para el handoff

### 2.1 Decisiones de diseño y alcance

- **Notebook canónico:** `Taller 4/juan_alvarez_daniel_ortiz.ipynb`. No crear un notebook paralelo como entregable principal: el enunciado pide **un** notebook con todas las celdas ejecutadas. Agregar secciones nuevas después de la celda 20, sin alterar el enunciado ni la solución de persona 1.
- **Artefactos nuevos:** `Taller 4/artefactos/persona2/`. No sobrescribir nada en `artefactos/persona1/`.
- **Documento de entrega nueva:** `Taller 4/artefactos/persona2/ENTREGA_PERSONA2.md`, consistente con el notebook y con `ENTREGA_PERSONA1.md`.
- **Idioma:** español, salvo nombres de funciones/rutas.
- **Principio de ejecución:** reutilizar `CNNReferencia`, `VAE`, `cargar_cnn_referencia`, `cargar_vae`, `cargar_ddpm`, `generar_vae`, `generar_ddpm`, `guardar_png_rgb`, `imagenes_a_tensor` y `mostrar_cuadricula` del notebook actual. No redefinir modelos ni reentrenar persona 1.
- **Orden clave:** evaluar VAE/DDPM → entrenar y evaluar GAN → decidir generador final → construir aumento sintético → entrenar CNN comparativas. Esto evita reentrenar la CNN de aumento si el bono cambia la elección.

### 2.2 Preparación del entorno (P0)

1. Desde la raíz del repo, verificar entorno. Si se decide usar el `.venv` existente, instalar dependencias faltantes con versiones compatibles con el torch ya presente:
   ```bash
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install diffusers==0.43.0 medmnist==3.0.2 \
     ipykernel nbformat nbconvert jupyter
   ```
   Si el entorno no resuelve dependencias, crear uno nuevo con Python 3.13.2 siguiendo las instrucciones exactas de `artefactos/persona1/ENTREGA_PERSONA1.md`, sección “Entorno y ejecución en VS Code”, e instalar torchvision compatible con el torch elegido desde el índice CUDA oficial correspondiente.
2. Confirmar:
   ```bash
   .venv/bin/python -c "import torch, torchvision, diffusers, medmnist, sklearn, PIL; print(torch.__version__, torch.cuda.is_available())"
   ```
3. Abrir o ejecutar siempre desde `Taller 4`.

### 2.3 Verificación de punto de partida (P0, continuación)

Antes de escribir código nuevo, ejecutar una celda de sanidad que verifique:

- `idx_sub` tiene longitud 200, valores únicos y `train.labels.ravel()[idx_sub] == 0`.
- `N == int((train.labels.ravel() == 0).sum()) == 852`.
- Los tres checkpoints cargan con `weights_only=True`.
- Existen 300 PNG en cada carpeta de generador, con `size == (28, 28)` y `mode == "RGB"`.
- Recalcular F1 macro, F1 basófilo y recall basófilo desde `metricas_cnn.json` y comparar con los valores de `configuracion.json` (tolerancia `1e-12`).

Si alguna verificación falla, detenerse: el problema está en la base heredada, no en las actividades pendientes.

### 2.4 Estructura de archivos a crear/modificar durante la ejecución

```
Taller 4/
├── juan_alvarez_daniel_ortiz.ipynb          # MODIFICAR: agregar secciones 3, 4, 5 y bono
├── PLAN_IMPLEMENTACION.md                   # Este archivo (ya creado)
└── artefactos/
    ├── persona1/                            # SOLO LECTURA
    └── persona2/                            # CREAR durante la ejecución
        ├── graficas/
        │   ├── muestras_test_basofilos.png
        │   ├── muestras_vae_eval.png
        │   ├── muestras_ddpm_eval.png
        │   ├── muestras_gan_eval.png
        │   ├── vecinos_vae_top5.png
        │   ├── vecinos_ddpm_top5.png
        │   ├── vecinos_gan_top5.png
        │   ├── matrices_confusion.png
        │   └── perdida_gan.png
        ├── imagenes_sinteticas_adicionales/
        │   ├── vae/                         # 352 PNG si se elige VAE
        │   ├── ddpm/                        # 352 PNG si se elige DDPM
        │   └── gan/                         # 352 PNG si se elige GAN
        ├── modelos/
        │   ├── cnn_sobremuestreo.pt
        │   ├── cnn_aumento_sintetico.pt
        │   └── gan/                         # generador/discriminador + metadatos
        ├── evaluacion/
        │   ├── metricas_generadores.json
        │   ├── predicciones_sinteticas.json
        │   └── metricas_clasificadores.json
        └── ENTREGA_PERSONA2.md
```

### 2.5 Interfaces y contratos de funciones nuevas

Agregar estas funciones en celdas nuevas del notebook, usando las ya definidas:

| Función | Contrato |
|---|---|
| `cargar_particiones()` | Devuelve `(train, val, test)` de `BloodMNIST(split=..., download=True)`. No reentrena nada. |
| `recuperar_indices(train)` | Carga `idx_sub.npy`, calcula `N` y `idx_otros`; hace `assert len(idx_sub)==200`, etiquetas 0 y `N==852`. Devuelve `(N, idx_sub, idx_otros)`. |
| `cargar_png_rgb(carpeta, cantidad=300)` | Devuelve `np.ndarray` uint8 NHWC en `[0,255]`, ordenado por nombre, validando 28×28×3. |
| `imagenes_a_float01(imagenes_uint8)` | Devuelve `np.ndarray` float32 NHWC en `[0,1]` para distancias. |
| `predecir_cnn(modelo, imagenes_uint8, device, batch_size=256)` | Aplica `imagenes_a_tensor(..., "cero_uno")`, usa `torch.no_grad()`, devuelve `np.ndarray` de clases `int64`. |
| `distancias_pares(x, y=None)` | Euclideanas sobre vectores aplanados. Si `y is None`, devuelve media de la triangular superior `i<j`; si no, devuelve media de todas las parejas `x-y`. |
| `vecinos_cercanos(consultas, referencia, n_top=5)` | Devuelve distancias mínimas por consulta, índices del vecino en `referencia` y los `n_top` pares más cercanos. |
| `entrenar_cnn(idx_basos, idx_sinteticos, nombre, semilla=42)` | Construye `idx_train = concatenate(idx_basos, idx_otros[, idx_sinteticos])`; reinicia semillas, crea `CNNReferencia()` nueva, entrena exactamente como la celda 12, guarda checkpoint en `modelos/{nombre}.pt` y devuelve `(modelo, historial, metricas, config)`. |
| `evaluar_clasificador(modelo, x_test, y_test)` | Devuelve F1 macro, F1 basófilo, recall basófilo y `confusion_matrix(labels=range(8))`. Usa `zero_division=0` para igualar la referencia. |
| `generar_adicionales(generador, cantidad=352)` | Usa `generar_vae`, `generar_ddpm` o el generador GAN, guarda PNG con `guardar_png_rgb` en la carpeta correspondiente y devuelve uint8 NHWC. |

Reglas comunes:

- Toda imagen sintética que entre a la CNN se convierte con `imagenes_a_tensor(..., "cero_uno")`.
- Toda imagen usada en distancias se convierte a `[0,1]` y se aplana a 2.352 dimensiones.
- Todas las rutas se derivan de `RAIZ = Path.cwd()` y `ARTEFACTOS_P2 = RAIZ / "artefactos" / "persona2"`.
- Los índices de `idx_sub` son relativos a `train`; nunca se re-sortean.

### 2.6 Actividad 3 — Evaluación de VAE y DDPM (P1)

**Entradas:** 300 PNG de `vae_imagenes/` y `ddpm_imagenes/`, `test.imgs`, `test.labels`, `cnn_referencia.pt`, los 200 basófilos de `train`.

**Pasos:**

1. **Calidad visual y consistencia**
   - Cuadrículas 4×4: 16 muestras VAE, 16 DDPM y 16 basófilos reales de `test` en orden oficial. Guardar en `graficas/`.
   - Clasificar las 300 sintéticas de cada modelo con la CNN de referencia.
   - Calcular `fraccion_basofilo = mean(pred == 0)` y compararla con `0.20081967213114754`.
   - Guardar conteo de las 8 clases predichas por modelo. Interpretar como consistencia con la clase, no como verdad biológica.

2. **Diversidad**
   - Seleccionar primeras 100 sintéticas por modelo y primeros 100 basófilos de `test`.
   - Calcular y reportar tres valores por modelo: media pairwise dentro de sintéticas, dentro de reales y entre sintéticas y reales.
   - La diversidad principal es la distancia media dentro de sintéticas; la comparación con reales es contextual.

3. **Posible memorización**
   - Para las 300 sintéticas de cada modelo, distancia mínima a los 200 basófilos de `train`.
   - Para los 244 basófilos de `test`, distancia mínima a esos mismos 200 ejemplos.
   - Reportar media, mediana, mínimo y percentiles 10/90 para ambos grupos.
   - Mostrar 5 pares sintético-entrenamiento más similares por modelo con sus distancias.

4. **Costo**
   - Registrar tiempos de muestreo heredados (0.0234 s VAE, 12.487 s DDPM) o medirlos de nuevo si se regenera; si se mide, identificar dispositivo y pasos de inferencia.

5. **Salida:** `evaluacion/metricas_generadores.json`, `evaluacion/predicciones_sinteticas.json` y gráficas.

**Esquema mínimo de `metricas_generadores.json`:**

```json
{
  "muestreo": {"n_visuales": 16, "n_distancia": 100, "regla": "primeras 100 PNG y primeros 100 basofilos de test"},
  "referencia": {"recall_basofilo_test": 0.20081967213114754},
  "modelos": {
    "VAE": {
      "clasificacion": {"n": 300, "fraccion_basofilo": 0.0, "conteo_predicciones": {}},
      "diversidad": {"media_sintetica": 0.0, "media_real": 0.0, "media_cruzada": 0.0},
      "memorizacion": {"sinteticas": {"media": 0.0, "mediana": 0.0, "min": 0.0}, "test": {"media": 0.0, "mediana": 0.0, "min": 0.0}, "top5": []},
      "costo": {"tiempo_generacion_300_s": 0.0, "pasos_inferencia": null}
    },
    "DDPM": {"...": "..."}
  }
}
```

### 2.7 Bono GAN (P2)

**Dependencia:** ejecutar después de la evaluación VAE/DDPM y antes de la decisión final de generador.

**Arquitectura/decisión:**

- Si se recupera el notebook del Laboratorio 10 referido por el enunciado, usar su GAN y documentar la procedencia.
- Si no está disponible, implementar una DCGAN compacta para 28×28 RGB:
  - Generador: `Linear(z_dim=64 -> 7*7*256)`, reshape, `ConvTranspose2d(256,128,4,2,1)+BatchNorm+ReLU`, `ConvTranspose2d(128,64,4,2,1)+BatchNorm+ReLU`, `Conv2d(64,3,3,padding=1)+Tanh`.
  - Discriminador: `Conv2d(3,64,3,2,1)+LeakyReLU(0.2)`, `Conv2d(64,128,3,2,1)+BatchNorm+LeakyReLU(0.2)`, Flatten, `Linear(7*7*128,1)`.
  - Pérdida: `BCEWithLogitsLoss` sobre logits del discriminador. Si se usa `Sigmoid+BCELoss` del laboratorio original, documentar el cambio.
  - Optimizadores: Adam para G y D, `lr=2e-4`, `betas=(0.5, 0.999)`, semilla 42.

**Entrenamiento:**

- Solo los 200 basófilos de `train`, RGB `[-1,1]`, `DataLoader` con lote 64 y generador de RNG sembrado.
- Una actualización de D y una de G por lote, o registrar la regla elegida si se difiere.
- Tope estricto de 1.200 s de GPU medidos con `torch.cuda.synchronize()`.
- Guardar `modelos/gan/generator.pt`, `discriminator.pt`, `metadata.json`/`.pt` con configuración, historial de pérdidas, épocas, actualizaciones y tiempo.
- Generar 300 PNG en `gan_imagenes/` con `000.png..299.png` y medir tiempo de muestreo.

**Evaluación equivalente a la actividad 3:**

- Misma cuadrícula de 16, clasificación de las 300 con la CNN, distancias sobre 100 muestras, vecinos más cercanos contra los 200 de entrenamiento y comparación con test.
- Evidencia de colapso de modos: duplicados exactos, distancia_media_sintética muy inferior a la de VAE/DDPM y a la real, concentración en pocos vecinos/predicciones, y cuadrícula visual monótona. Reportar números, no solo impresión visual.
- Actualizar la comparación con VAE y DDPM y responder si cambia la elección del generador.

### 2.8 Decisión del generador (P3)

Antes del aumento sintético, escribir una celda/markdown con una tabla de decisión. Criterios obligatorios:

1. Consistencia: fracción clasificada como basófilo, comparada contra el recall de referencia.
2. Diversidad: distancia media dentro de 100 sintéticas, comparada contra 100 basófilos reales.
3. Memorización: distancia al vecino más cercano de `train`, comparada contra basófilos de `test`.
4. Calidad visual y defectos observados en cuadrículas.
5. Costo de entrenamiento/generación y complejidad.
6. Para GAN, evidencia de colapso de modos.

La decisión debe quedar registrada en `metricas_generadores.json` como:

```json
{"seleccion": {"generador": "VAE|DDPM|GAN", "justificacion": "texto", "cambia_por_bono": false}}
```

No usar lenguaje de privacidad o clínica como conclusión; decir explícitamente que las métricas son académicas y limitadas.

### 2.9 Actividad 4 — Clasificadores (P4)

**Conjuntos:**

| Modelo | Índices de basófilos en `train` | Total train |
|---|---|---:|
| Referencia | `idx_sub` (200) | 11.307 |
| Sobremuestreo | `np.resize(idx_sub, N)` (852) | 11.959 |
| Aumento sintético | `idx_sub` (200) + 652 sintéticos del generador final | 11.959 |

En todos los casos se incluyen los mismos 11.107 índices de otras clases (`idx_otros`) y se mantienen `val` y `test` oficiales intactos.

**Sintéticas adicionales:**

- Reutilizar las 300 PNG ya generadas por el modelo elegido.
- Generar 352 adicionales con la misma función, mismos pasos de inferencia si aplica y semilla documentada distinta; guardarlas en `imagenes_sinteticas_adicionales/<modelo>/`.
- Concatenar 300 + 652 en ese orden; validar forma `(652, 28, 28, 3)` y rango uint8.

**Entrenamiento:**

- Reutilizar `CNNReferencia` sin cambios.
- Adam `lr=1e-3`, lote 128, máximo 10 épocas, `CrossEntropyLoss`, EarlyStopping sobre `val_loss` con paciencia 3 y restauración de mejores pesos.
- Reiniciar `torch.manual_seed(42)`, semillas CUDA y `torch.Generator().manual_seed(42)` para el `DataLoader` antes de **cada** entrenamiento.
- Guardar checkpoints con las mismas claves que la referencia: `state_dict`, `historial`, `epocas`, `config`. Agregar en `config` el campo `generador` y la semilla.

**Evaluación y salidas:**

- Evaluar los tres modelos sobre el mismo `test` de 3.421 imágenes.
- Tabla con filas `Referencia`, `Sobremuestreo`, `Aumento sintético` y columnas F1 macro, F1 basófilo, recall basófilo.
- Matrices de confusión con `labels=list(range(8))` y resaltado; guardar `graficas/matrices_confusion.png`.
- `evaluacion/metricas_clasificadores.json` con métricas, matrices, `y_true`, `y_pred` e historiales.

### 2.10 Actividad 5 — Análisis y entrega (P5)

En celdas markdown, responder:

1. Comparación VAE vs DDPM (y GAN si se hace el bono): calidad visual, diversidad, memorización, costo, relación con el funcionamiento de cada modelo y límites de las métricas, incluido que la CNN evaluadora compartió datos con los generadores.
2. Utilidad del aumento sintético frente a real y frente a sobremuestreo; relación entre métricas de imagen y resultados de clasificación; cautela por una sola semilla y por la posibilidad de que las sintéticas no aporten información nueva.
3. Nota al laboratorio de **máximo 300 palabras**: recomendación sobre complementar datos y sobre compartir sintéticas en lugar de reales, justificación, limitaciones y evidencia adicional necesaria antes de cualquier uso clínico.

La nota debe incluir, como mínimo: resultados numéricos de las tres CNN, decisión de generador, principales limitaciones, licencia CC BY 4.0 y referencias de MedMNIST/diffusers/GAN.

### 2.11 Empaquetado final (P6)

1. Escribir `artefactos/persona2/ENTREGA_PERSONA2.md` con el mismo nivel de detalle que `ENTREGA_PERSONA1.md`, apuntando a rutas relativas y a los valores finales reales obtenidos.
2. Ejecutar el notebook completo desde cero en el entorno elegido:
   ```bash
   cd "Taller 4"
   ../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
     juan_alvarez_daniel_ortiz.ipynb --ExecutePreprocessor.timeout=3600
   ```
   Alternativa en VS Code: seleccionar kernel, confirmar directorio `Taller 4` y ejecutar “Restart Kernel and Run All Cells”.
3. Verificar que todas las celdas de código quedan con `execution_count` no nulo y sin errores en `outputs`.
4. Revisar `git status`: solo deben aparecer el notebook modificado, `PLAN_IMPLEMENTACION.md` y los archivos nuevos bajo `artefactos/persona2/`.
5. No hacer commit salvo solicitud explícita.

---

## 3. Estrategia de verificación y pruebas

### 3.1 Pruebas de datos y artifacts

| Verificación | Criterio de aceptación |
|---|---|
| Índices | `len(idx_sub)==200`, únicos, etiqueta 0, `N==852`. |
| Particiones | Conteos por clase idénticos a la tabla de `ENTREGA_PERSONA1.md`. |
| Checkpoints | Cargan con `weights_only=True`; salida de CNN para un lote es `(n, 8)`; VAE decodifica a `(n,3,28,28)`. |
| PNG | 300 por generador base, 300 para GAN si se hace el bono; 352 adicionales si aplica; todos RGB 28×28. |
| Sin filtración | Ningún índice de `val`/`test` aparece en los conjuntos de entrenamiento de CNN; las sintéticas se generan solo desde modelo entrenado con los 200 de `train`. |

### 3.2 Pruebas de métricas

- `fraccion_basofilo` debe estar en `[0,1]`; el conteo de predicciones debe sumar 300.
- Todas las distancias deben ser finitas y `>= 0`; la distancia media dentro de sintéticas debe simétrica e independiente del orden.
- `nn_test` debe calcularse sobre 244 basófilos de `test`, no sobre 100.
- Los `top5` deben ordenarse de menor a mayor distancia y contener índices válidos en `[0,199]`.
- Las matrices de confusión deben tener forma `(8,8)` y la suma de filas debe igualar el conteo real por clase en `test` (`[244,624,311,579,243,284,666,470]`).
- Las métricas de la CNN de referencia recalculadas desde `metricas_cnn.json` deben coincidir con las del JSON de configuración.

### 3.3 Pruebas de entrenamiento

- Comparar el historial de la CNN de referencia con el esperado: 10 épocas, mejor `val_loss` `0.472257248709135`.
- Para cada CNN nueva, verificar que `config.semilla == 42`, arquitectura idéntica y que el checkpoint puede recargarse con `CNNReferencia`.
- Si la ejecución se repite en el mismo entorno/dispositivo, las dos CNN nuevas deberían dar métricas idénticas o con diferencias numéricas menores a tolerancia documentada; en caso de diferencia, atribuirla a no determinismo de CUDA y documentarlo.

### 3.4 Pruebas del bono

- `metadata` de la GAN debe mostrar tiempo de entrenamiento `<= 1200` s de GPU.
- Debe existir `gan_imagenes/000.png..299.png`.
- El JSON de evaluación debe contener para GAN los mismos bloques que VAE/DDPM: clasificación, diversidad, memorización y costo.
- La discusión de colapso debe incluir al menos dos evidencias cuantitativas y una visual.

### 3.5 Pruebas de entrega

- Notebook: celdas ejecutadas, sin errores, secciones ordenadas como Actividad 3, Bono, Actividad 4, Actividad 5 (o Bono después de Actividad 3 y antes de la decisión final).
- Nota al laboratorio: verificar longitud con un conteo de palabras; debe ser `<= 300`.
- Referencias y licencias presentes.
- `ENTREGA_PERSONA2.md` consistente con los JSON finales.

---

## 4. Riesgos y puntos de atención durante la ejecución

- Si `diffusers` o `medmnist` no pueden instalarse en el entorno actual, no intentar parchear el notebook: crear un entorno limpio con las versiones documentadas.
- Si la GPU se queda sin memoria con DDPM o GAN, bajar `batch_size` de inferencia/gan y documentarlo; no cambiar la arquitectura de la CNN.
- No confundir los 1.000 pasos de difusión con actualizaciones de parámetros (2.100 reales).
- No usar `val` para entrenar ni para seleccionar el generador; `val` solo se usa en EarlyStopping, como en la actividad 1.
- No presentar el recall de la CNN sobre sintéticas como prueba de realismo: la CNN vio esos 200 ejemplos reales durante el entrenamiento.
- No afirmar que las imágenes sintéticas protegen la privacidad sin estudios de privacidad; el enunciado solo pide “posible memorización”.
- Si el GAN no termina dentro del presupuesto, conservar checkpoint y reportar el tiempo; no ocultar la interrupción.
- Si el generador elegido por el bono difiere del provisional, regenerar las 352 adicionales con ese modelo y actualizar toda la cadena de artefactos antes de entrenar la CNN de aumento.

---

## 5. Checklist final para la nueva sesión

- [ ] Entorno reproducible verificado.
- [ ] Sanidad de persona 1 verificada sin reentrenar.
- [ ] Actividad 3 ejecutada para VAE y DDPM con JSON, tablas y gráficas.
- [ ] Bono GAN entrenado dentro de 20 min de GPU y evaluado con el mismo protocolo.
- [ ] Generador final seleccionado y justificado por escrito.
- [ ] 300 + 352 sintéticas del generador final generadas y validadas.
- [ ] CNN de sobremuestreo y de aumento sintético entrenadas con semilla 42 y configuración idéntica.
- [ ] Tabla de tres clasificadores y matrices de confusión generadas.
- [ ] Análisis de la actividad 5 y nota ≤300 palabras escritos.
- [ ] Notebook completo ejecutado de arriba abajo sin errores.
- [ ] `ENTREGA_PERSONA2.md` creado y consistente.
- [ ] `git status` revisado; sin commits no solicitados.
