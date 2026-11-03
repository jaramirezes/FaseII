# 🚗 Sistema Inteligente de Evaluación de Siniestros y Daños Vehiculares (Fase II)

Solución técnica completa, contenerizada y desplegable orientada al procesamiento y clasificación automatizada de daños en vehículos mediante modelos de Deep Learning (Vision Transformer - ViT), integrada a través de una API funcional.

\---

## 📋 Tabla de Contenidos

1. [Descripción General del Proyecto](#-descripción-general-del-proyecto)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Contenedor Docker Funcional](#-contenedor-docker-funcional)
4. \[Manual de Despliegue en la Nube](#-manual-de-despliegue-en-la- nube)
5. [Documento de Validación y Pruebas](#-documento-de-validación-y-pruebas)

\---

## 🎯 1. Descripción General del Proyecto

Este proyecto implementa una arquitectura backend robusta para la evaluación automatizada de siniestros vehiculares. Utiliza técnicas de visión por computadora con modelos preentrenados (*Vision Transformer*) para clasificar el estado y tipo de daño en unidades vehiculares a partir de un conjunto de datos especializado de imágenes. La aplicación expone endpoints funcionales mediante una API desarrollada en Python para facilitar su integración con interfaces de usuario o sistemas empresariales externos.

\---

## 📂 2. Estructura del Repositorio

El repositorio se encuentra organizado de la siguiente manera para garantizar portabilidad y orden técnico:

```text
FaseII/
.

├── carga/                                    # Carpeta de insumos y catálogos base

│   ├── imagenes\_entrenamiento.zip            # Dataset comprimido para entrenamiento

│   └── tabulador\_costos\_siniestros.csv       # Tabulador financiero de costos de siniestros

├── dataset\_vehiculos/                        # Imágenes limpias y listas para evaluación

│   ├── 1\_jpeg.rf.e50d9b86c2a17747f...jpg     # Archivo de imagen de prueba de daños

│   ├── 1\_jpeg.rf.6cc41989e29fc638...jpg      # Archivo de imagen de prueba de daños

│   ├── 6---1--jpeg.rf.81ff429e14d...jpg      # Archivo de imagen de prueba de daños

│   ├── 11\_jpeg.rf.2a13b0d994e1a1d...jpg      # Archivo de imagen de prueba de daños

│   ├── 15b654e78bcf27d9d8fcec49...jpg        # Archivo de imagen de prueba de daños

│   ├── 17\_jpeg.rf.482e834e083b0c3...jpg      # Archivo de imagen de prueba de daños

│   ├── 22\_jpeg.rf.11b25fd9d44b2f4...jpg      # Archivo de imagen de prueba de daños

│   └── ...                                   # Resto de imágenes del dataset de evaluación

├── modelo\_vit\_siniestros/                    # Modelo Vision Transformer (ViT) entrenado

│   ├── config.json                           # Archivo de configuración de la arquitectura ViT

│   ├── model.safetensors                     # Pesos del modelo optimizados con SafeTensors

│   └── preprocessor\_config.json              # Configuración del procesador de imágenes

├── Dockerfile                                # Configuración para la contenerización local

├── main.py                                   # Script de ejecución general del sistema

├── modelo\_siniestros.py                      # Módulo principal con la lógica del modelo y flujos

├── README.md                                 # Documentación del proyecto

├── requirements.txt                          # Archivo de dependencias y librerías Python

└── simular.py                                # Script de simulación, inferencia y cruce con tabulador

