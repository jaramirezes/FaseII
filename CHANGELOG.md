# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## 1.0.0 - 2026-11-08

### Añadido
- **Estructura del Proyecto**: Creación de la carpeta `fuentes/` para centralizar todos los módulos lógicos y scripts ejecutables (`main.py`, `modelo_siniestros.py`, `simular.py` y `frontend.py`).
- **Gestión de Datos (`datos/`)**: Organización de los insumos (`imagenes_entrenamiento.zip` y `tabulador_costos_siniestros.csv`) dentro del directorio unificado de datos.
- **Modelo Vision Transformer (ViT)**: Integración del modelo entrenado `modelo_vit_siniestros` para clasificación de daños y cálculo de costos financieros.
- **Contenerización y Despliegue**: Configuración de `Dockerfile` y `requirements.txt` para la ejecución modular y reproducible del entorno.