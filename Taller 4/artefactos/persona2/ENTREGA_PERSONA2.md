# Entrega de la persona 2 — Taller 4

## Estado y entorno

Completadas: actividad 3 (evaluación de generadores), bono GAN, decisión de generador, actividad 4 (tres CNN), actividad 5 (análisis y nota) y empaquetado. No se reentrenaron la CNN, el VAE ni el DDPM de persona 1: se cargaron sus checkpoints y se reutilizaron sus 300 PNG por modelo.

Ejecución: Python 3.14.10, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, diffusers 0.43.0, MedMNIST 3.0.2, dispositivo cuda (NVIDIA GeForce RTX 2070).
Directorio de trabajo: `Taller 4` (el notebook usa `Path.cwd()`). Para preservar `artefactos/persona1/`, las gráficas y recálculos de las celdas heredadas se redirigieron a `artefactos/persona2/auditoria/`; `idx_sub.npy` solo se verificó.

## Actividad 3 y bono GAN

Reglas: primeras 100 PNG y primeros 100 basófilos de test para distancias; 300 sintéticas para clasificación y vecinos; 244 basófilos de test frente a los 200 de train para comparación de memorización. Recall de referencia: 0.20081967213114754.

| Generador | Fracción basófila | Diversidad sintética | Diversidad real | Cruzada | NN sintética media | NN test media | Tiempo 300 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE | 0.030 | 6.495 | 8.194 | 7.431 | 4.882 | 5.766 | 0.023 |
| DDPM | 0.030 | 7.521 | 8.194 | 8.457 | 5.977 | 5.766 | 12.487 |
| GAN | 0.073 | 7.863 | 8.194 | 8.123 | 5.827 | 5.766 | 0.262 |

Decisión: **GAN** (provisional DDPM, cambia por el bono=True). Justificación completa en `evaluacion/metricas_generadores.json`, campo `seleccion`.
GAN DCGAN adaptada del Laboratorio 10 (inicialización N(0,0.02), snapshots con ruido fijo): 600 épocas, 2400 actualizaciones, 61.634 s de GPU (límite 1200 s, cumplido=True); Adam lr=2e-4, betas=(0.5,0.999), lote 64, semilla 42, etiquetas reales 0.9. Checkpoints e instantáneas en `modelos/gan/`.
Indicadores de colapso aplicados a los tres modelos: {"VAE": {"fraccion_duplicados_exactos": 0.0, "concentracion_vecino_train": 0.05333333333333334, "concentracion_clase_predicha": 0.8333333333333334, "distancia_media_sintetica": 6.495081424713135, "ratio_diversidad_vs_mejor_base": 0.8636039769532697, "indicios_colapso": true}, "DDPM": {"fraccion_duplicados_exactos": 0.0, "concentracion_vecino_train": 0.11333333333333333, "concentracion_clase_predicha": 0.5266666666666666, "distancia_media_sintetica": 7.520902633666992, "ratio_diversidad_vs_mejor_base": 1.0, "indicios_colapso": false}, "GAN": {"fraccion_duplicados_exactos": 0.0, "concentracion_vecino_train": 0.056666666666666664, "concentracion_clase_predicha": 0.73, "distancia_media_sintetica": 7.863175392150879, "ratio_diversidad_vs_mejor_base": 1.045509531921304, "indicios_colapso": false}}.

## Actividad 4

Generador elegido GAN: 300 PNG reutilizadas + 352 adicionales (semilla 1303) en `imagenes_sinteticas_adicionales/gan/`. Conjuntos: 11.307 / 11.959 / 11.959 imágenes. CNN idénticas, Adam 1e-3, lote 128, 10 épocas máximo, EarlyStopping val_loss patience=3 con restauración, semilla 42 reiniciada por entrenamiento.

| Clasificador | F1 macro | F1 basófilo | Recall basófilo |
|---|---:|---:|---:|
| Referencia | 0.769993 | 0.331081 | 0.200820 |
| Sobremuestreo | 0.823533 | 0.680761 | 0.659836 |
| Aumento sintético | 0.823654 | 0.672606 | 0.618852 |

Matrices de confusión (8,8), y_true, y_pred e historiales: `evaluacion/metricas_clasificadores.json`; sumas de fila verificadas: [244, 624, 311, 579, 243, 284, 666, 470].

## Archivos

- `graficas/`: cuadrículas de evaluación, vecinos top 5, matrices de confusión, pérdida GAN y evolución GAN con ruido fijo.
- `gan_imagenes/000.png..299.png` e `imagenes_sinteticas_adicionales/<generador>/000.png..351.png`.
- `modelos/cnn_sobremuestreo.pt`, `modelos/cnn_aumento_sintetico.pt`, `modelos/gan/`.
- `evaluacion/metricas_generadores.json`, `predicciones_sinteticas.json`, `metricas_clasificadores.json`, `nota_laboratorio.md`.
- `auditoria/`: gráficas y recálculos de persona 1 redirigidos para no sobrescribir sus entregables.

## Nota al laboratorio

Recomendamos complementar el entrenamiento con imágenes sintéticas solo como experimento controlado, no como sustituto validado de datos reales. En el test oficial de 3,421 imágenes, la CNN de referencia alcanzó F1 macro 0.769993, F1 de basófilo 0.331081 y recall 0.200820; con 652 sintéticas GAN mejoró a 0.823654/0.672606/0.618852, mientras que el sobremuestreo de los 200 basófilos reales dio 0.823533/0.680761/0.659836 y conservó mejor F1 y recall de basófilo. Por tanto, conviene conservar las reales y tratar las generadas como apoyo, no como reemplazo al compartir entre instituciones.

Se eligió GAN porque su diversidad (7.863 frente a 8.194 en 100 basófilos reales), su distancia al vecino de entrenamiento (5.827 frente a 5.766 en test) y su consistencia superaron a las alternativas según la ponderación documentada. Indicadores finales: {"VAE": true, "DDPM": false, "GAN": false}; el generador elegido no mostró indicios de colapso. Solo 7.3% de sus sintéticas fue clasificado como basófilo, frente al recall 20.1% en reales.

Limitaciones: una sola semilla, 200 basófilos de entrenamiento, imágenes de 28×28, CNN evaluadora entrenada con esos mismos 200 ejemplos y métricas de píxeles que no miden verosimilitud biológica ni privacidad. Se requieren múltiples semillas, revisión por expertos, validación externa multiinstitucional, pruebas de privacidad y evaluación clínica antes de cualquier uso asistencial. BloodMNIST es CC BY 4.0 (código MedMNIST Apache-2.0); citar MedMNIST v2, Ho et al., Goodfellow et al. y Radford et al.

## Referencias y límites

BloodMNIST/MedMNIST: CC BY 4.0 para los datos; el código MedMNIST es Apache-2.0. Yang et al., *MedMNIST v2*, Scientific Data 10, 41 (2023), doi:10.1038/s41597-022-01721-8; Yang, Shi & Ni, ISBI 2021, doi:10.1109/ISBI48211.2021.9434062. DDPM con Hugging Face diffusers y Ho et al., NeurIPS 2020, arXiv:2006.11239. GAN: Goodfellow et al., NeurIPS 2014, y patrón DCGAN (Radford et al., ICLR 2016).
Resultados académicos de una sola semilla y 200 basófilos; la CNN evaluadora compartió datos con los generadores. No se hacen afirmaciones clínicas ni de privacidad.

