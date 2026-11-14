# Entrega persona 1 — Taller 4

## Estado

Completadas: actividad 1 (exploración y CNN de referencia), actividad 2.1 (VAE), actividad 2.2 (DDPM) y actividad 2.3 (300 PNG por generador). Las actividades 3, 4, 5 y el bono GAN quedan explícitamente para la persona 2.

Dispositivo de ejecución: cuda (NVIDIA GeForce RTX 3060 Ti)
Python 3.13.2; PyTorch 2.14.1+cu130; MedMNIST 3.0.2; diffusers 0.43.0.

## Datos y CNN de referencia

BloodMNIST oficial: N=852 basófilos en train; índices únicos seleccionados con `default_rng(42)`: 200. Conteos train: `{'Basófilo': 852, 'Eosinófilo': 2181, 'Eritroblasto': 1085, 'Granulocito inmaduro': 2026, 'Linfocito': 849, 'Monocito': 993, 'Neutrófilo': 2330, 'Plaqueta': 1643}`; val: `{'Basófilo': 122, 'Eosinófilo': 312, 'Eritroblasto': 155, 'Granulocito inmaduro': 290, 'Linfocito': 122, 'Monocito': 143, 'Neutrófilo': 333, 'Plaqueta': 235}`; test: `{'Basófilo': 244, 'Eosinófilo': 624, 'Eritroblasto': 311, 'Granulocito inmaduro': 579, 'Linfocito': 243, 'Monocito': 284, 'Neutrófilo': 666, 'Plaqueta': 470}`.
CNN: 10 épocas, mejor val_loss=0.472257; test F1 macro=0.769993, F1 basófilo=0.331081, recall basófilo=0.200820.
Entrada CNN RGB en [0,1]. Checkpoint: `artefactos/persona1/cnn_referencia.pt`; métricas, `y_true`, `y_pred` e historial: `artefactos/persona1/metricas_cnn.json`.
El archivo `idx_sub.npy` contiene las posiciones dentro de `train`; los 200 basófilos originales se reconstruyen con `train.imgs[np.load(ARTEFACTOS / 'idx_sub.npy')]`. Recuperar N: `int((train.labels.ravel() == 0).sum())`. La persona 2 puede reusar `CNNReferencia` del notebook y cargar `state_dict` del checkpoint; sus datos sintéticos deben convertirse con el mismo `imagenes_a_tensor(..., 'cero_uno')`.

## Generadores

VAE: latent_dim=16, Adam lr=1e-3, batch=32, 300 épocas; pérdidas finales total=49.780727, reconstrucción=28.089805, KL=21.690921; entrenamiento 13.07s (GPU). Checkpoint: `artefactos/persona1/vae.pt`.
DDPM: UNet2DModel canales [64, 128], DDPMScheduler train=1000, AdamW lr=1e-4, batch=32; 300 épocas completas, parcial=False, 2100 actualizaciones; MSE final=0.030810; entrenamiento 112.04s (GPU). Checkpoint y configuración: `artefactos/persona1/ddpm/`.
Los generadores reciben RGB `[0,255] -> [-1,1]`; al guardar se mapea a PNG RGB `[0,255]`. VAE 300 imágenes: `artefactos/persona1/vae_imagenes/`; DDPM 300 imágenes: `artefactos/persona1/ddpm_imagenes/`, inferencia=50 pasos. Tiempo de muestreo de 300: VAE 0.02s; DDPM 12.49s.
Funciones para generar cantidades arbitrarias: `generar_vae(modelo, cantidad, ...)` y `generar_ddpm(modelo, cantidad, pasos_inferencia=50, ...)`. Carga VAE: `modelo_vae, vae_checkpoint = cargar_vae()`. Carga DDPM sin reentrenar: `modelo_ddpm, scheduler_ddpm, metadata = cargar_ddpm()`.

## Continuación de la persona 2

Al reabrir el notebook, ejecutar hasta las celdas que definen las clases y funciones de carga; el código detecta checkpoints y no vuelve a entrenar. Para la actividad 3, inspeccionar 16 muestras y comparar las 300 imágenes con test usando los procedimientos del enunciado. Para la actividad 4, `N-200` imágenes sintéticas se pueden pedir al generador elegido mediante `generar_vae` o `generar_ddpm`; reconstruir las imágenes de train y test desde `BloodMNIST(split=..., download=True)`, conservar particiones oficiales y evaluar las tres CNN. La persona 2 debe ejecutar por separado las actividades 3, 4 y 5: aquí no se calcularon distancias/diversidad/memorización, no se eligió generador ni se entrenaron las dos CNN comparativas.

## Reproducción, fuentes y limitaciones

Entorno usado: Python 3.13.2 y el `.venv` del repositorio. Activar en PowerShell: `.venv\Scripts\Activate.ps1`. Instalar PyTorch para CUDA desde la instrucción oficial compatible con el driver/GPU y después `python -m pip install diffusers medmnist numpy pandas matplotlib scikit-learn pillow ipykernel jupyter nbformat nbconvert`. Abrir `Taller 4/juan_alvarez_daniel_ortiz.ipynb` en VS Code, seleccionar `.venv` como kernel y ejecutar de arriba abajo. MedMNIST descarga BloodMNIST a su caché de usuario; no se copia el dataset al repositorio.
MedMNIST informa que BloodMNIST está bajo CC BY 4.0 (dataset); citar Yang et al., *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*, Scientific Data 10, 41 (2023), doi:10.1038/s41597-022-01721-8, y también Yang, Shi & Ni, ISBI 2021, doi:10.1109/ISBI48211.2021.9434062. La licencia del código MedMNIST es Apache-2.0 y es distinta de la licencia de los datos. DDPM se implementó con Hugging Face Diffusers; citar Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) y la documentación de diffusers. BloodMNIST y estos resultados son académicos; no son para uso clínico.
Fuentes oficiales: https://medmnist.com/ ; https://github.com/MedMNIST/MedMNIST ; https://doi.org/10.5281/zenodo.10519652 ; https://huggingface.co/docs/diffusers/ ; https://arxiv.org/abs/2006.11239 .

## Evidencia de inspecci?n visual

Las cuadr?culas de 48 im?genes reales y las muestras sint?ticas se conservaron en `artefactos/persona1/graficas/`. La interpretaci?n visual se redacta despu?s de inspeccionar esas figuras; la calidad aparente no sustituye la evaluaci?n formal que corresponde a la actividad 3.
