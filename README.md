# 🚗 Sistema Inteligente de Evaluación de Siniestros y Daños Vehiculares (Fase II)

Solución técnica completa, contenerizada y desplegable orientada al procesamiento y clasificación automatizada de daños en vehículos mediante modelos de Deep Learning (Vision Transformer - ViT), integrada a través de una API funcional.


## 📋 Tabla de Contenidos

1. Descripción General del Proyecto(#-descripción-general-del-proyecto)
2. Estructura del Repositorio(#-estructura-del-repositorio)
3. Contenedor Docker Funcional(#-contenedor-docker-funcional)
4. Manual de Despliegue en la Nube(#-manual-de-despliegue-en-la- nube)
5. Documento de Validación y Pruebas(#-documento-de-validación-y-pruebas)

## 🎯 1. Descripción General del Proyecto

Este proyecto implementa una arquitectura backend robusta para la evaluación automatizada de siniestros vehiculares. Utiliza técnicas de visión por computadora con modelos preentrenados (*Vision Transformer*) para clasificar el estado y tipo de daño en unidades vehiculares a partir de un conjunto de datos especializado de imágenes. La aplicación expone endpoints funcionales mediante una API desarrollada en Python para facilitar su integración con interfaces de usuario o sistemas empresariales externos.

## 📂 2. Estructura del Repositorio

El repositorio se encuentra organizado de la siguiente manera para garantizar portabilidad y orden técnico:

```text
FaseII
.

├── carga                                     # Carpeta de insumos y catálogos base
│   ├── imagenes_entrenamiento.zip            # Dataset comprimido para entrenamiento
│   └── tabulador_costos_siniestros.csv       # Tabulador financiero de costos de siniestros
├── dataset_vehiculos                         # Imágenes limpias y listas para evaluación
│   ├── 1_jpeg.rf.e50d9b86c2a17747f...jpg     # Archivo de imagen de prueba de daños
│   ├── 1_jpeg.rf.6cc41989e29fc638...jpg      # Archivo de imagen de prueba de daños
│   ├── 6---1--jpeg.rf.81ff429e14d...jpg      # Archivo de imagen de prueba de daños
│   └── ...                                   # Resto de imágenes del dataset de evaluación
├── modelo_vit_siniestros                     # Modelo Vision Transformer (ViT) entrenado
│   ├── config.json                           # Archivo de configuración de la arquitectura ViT
│   ├── model.safetensors                     # Pesos del modelo optimizados con SafeTensors (No lo cargue por que mide 335 mb, se anexa en una liga de drive)
│   └── preprocessor_config.json              # Configuración del procesador de imágenes
├── Dockerfile                                # Configuración para la contenerización local
├── frontend.py                               # Interfaz de usuario interactiva desarrollada con Streamlit para el consumo de la API.
├── main.py                                   # Script de ejecución general del sistema para generar la API.
├── modelo_siniestros.py                      # Módulo principal con la lógica del modelo y flujos
├── modelo\\\_siniestros.py                      # Módulo principal con la lógica del modelo y flujos
├── README.md                                 # Documentación del proyecto
├── requirements.txt                          # Archivo de dependencias y librerías Python
└── simular.py                                # Script de simulación, inferencia y cruce con tabulador


