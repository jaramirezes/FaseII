# 🚗 Sistema Inteligente de Evaluación de Siniestros y Daños Vehiculares (Fase II)

Solución técnica completa, contenerizada y desplegable orientada al procesamiento y clasificación automatizada de daños en vehículos mediante modelos de Deep Learning (Vision Transformer - ViT), integrada a través de una API funcional.


## 📋 Tabla de Contenidos

1. Descripción General del Proyecto
2. Estructura del Repositorio
3. [Contenedor Docker Funcional](docs/Contenedor%20Docker%20funcional.docx)
4. [Manual de Despliegue en la Nube](docs/Manual%20de%20despliegue%20en%20la%20nube.docx)
5. [Documento de Validación y Pruebas](docs/Validaci%C3%B3n%20y%20Pruebas.docx)

## 🎯 1. Descripción General del Proyecto

Este proyecto implementa una arquitectura backend robusta para la evaluación automatizada de siniestros vehiculares. Utiliza técnicas de visión por computadora con modelos preentrenados (*Vision Transformer*) para clasificar el estado y tipo de daño en unidades vehiculares a partir de un conjunto de datos especializado de imágenes. La aplicación expone endpoints funcionales mediante una API desarrollada en Python para facilitar su integración con interfaces de usuario o sistemas empresariales externos.

## 📂 2. Estructura del Repositorio

El repositorio se encuentra organizado de la siguiente manera para garantizar portabilidad y orden técnico:

```text
FaseII
.

├── datos                                     # Carpeta de insumos y catálogos base
│   ├── imagenes_entrenamiento.zip            # Dataset comprimido para entrenamiento
│   └── tabulador_costos_siniestros.csv       # Tabulador financiero de costos de siniestros
├── docs/                                     # Documentos del proyecto
│   ├── Contenedor Docker Funcional.docx      # Guía y evidencias de la configuración y ejecución del contenedor Docker
│   ├── Manual de Despliegue en la Nube.docx  # Instrucciones paso a paso para el despliegue del servicio en la nube
│   ├── FaseII.docx                           # Reporte integrador del proyecto 
│   └── Validación y Pruebas.docx             # Casos de prueba, script de pruebas funcionales y simulación de peticiones
├── fuentes/                                  # Módulos y scripts de código fuente del sistema
│   ├── frontend.py                           # Interfaz gráfica y de usuario para interacción
│   ├── main.py                               # Script principal de la API y puntos de entrada (Endpoints)
│   ├── modelo_siniestros.py                  # Lógica de procesamiento y arquitectura de inferencia
│   └── simular.py                            # Script de pruebas funcionales y simulación de peticiones
├── modelo_vit_siniestros                     # Modelo Vision Transformer (ViT) entrenado
│   ├── config.json                           # Archivo de configuración de la arquitectura ViT
│   ├── model.safetensors                     # Pesos del modelo optimizados con SafeTensors (No lo cargue por que mide 335 mb, se anexa en una liga de drive)
│   └── preprocessor_config.json              # Configuración del procesador de imágenes
├── Dockerfile                                # Configuración para la contenerización local
├── README.md                                 # Documentación del proyecto
└── requirements.txt                          # Archivo de dependencias y librerías Python
 

