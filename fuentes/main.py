import io
import os
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(
    title="API de Evaluación de Siniestros Automotrices",
    version="1.0.0",
    description="Backend para la predicción de daños con ViT y cálculo de montos según tabulador.",
)

# Variables globales para los recursos
v_modelo_vit = None
v_image_processor = None
v_df_tabulador = None
v_clases_dano = []
v_device = None

@app.on_event("startup")
def cargar_recursos():
    global v_modelo_vit, v_image_processor, v_df_tabulador, v_clases_dano, v_device
    
    v_directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Cargar el tabulador de costos
    v_ruta_tabulador = os.path.join(v_directorio_actual, "carga", "tabulador_costos_siniestros.csv")

    if os.path.exists(v_ruta_tabulador):
        v_df_tabulador = pd.read_csv(v_ruta_tabulador)
        print("Tabulador de costos cargado exitosamente en la API.")
    else:
        print("No se encontró el dataset tabulador de costos.")

    # 2. Cargar el modelo ViT entrenado localmente
    v_directorio_modelo = os.path.join(v_directorio_actual, "modelo_vit_siniestros")
    print("Cargando modelo ViT y procesador en FastAPI...")
    
    v_image_processor = AutoImageProcessor.from_pretrained(v_directorio_modelo, local_files_only=True)
    v_modelo_vit = AutoModelForImageClassification.from_pretrained(v_directorio_modelo, local_files_only=True)

    # Extraer las clases de daño dinámicamente desde la configuración del modelo
    v_id2label = v_modelo_vit.config.id2label
    v_clases_dano = [v_id2label[i] for i in sorted(v_id2label.keys())]

    v_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_modelo_vit.to(v_device)
    v_modelo_vit.eval()
    print("Recursos de IA cargados correctamente en memoria.")

@app.get("/")
def estado_api():
    return {"estado": "OK", "mensaje": "API de Evaluación de Siniestros Operativa"}

@app.post("/predict")
async def evaluar_siniestro(
    marca: str = Form(...),
    modelo: str = Form(...),
    anio: int = Form(...),
    tipo_vehiculo: str = Form(...),
    imagen: UploadFile = File(...)
):
    try:
        # Leer y procesar la imagen enviada por HTTP
        v_bytes_imagen = await imagen.read()
        v_image = Image.open(io.BytesIO(v_bytes_imagen)).convert("RGB")
        
        v_inputs = v_image_processor(images=v_image, return_tensors="pt")
        v_pixel_values = v_inputs["pixel_values"].to(v_device)

        # Inferencia con el modelo ViT
        with torch.no_grad():
            v_outputs = v_modelo_vit(pixel_values=v_pixel_values)
            v_logits = v_outputs.logits
            v_probabilidades = torch.nn.functional.softmax(v_logits, dim=-1)[0]
            v_idx = v_logits.argmax(-1).item()
            v_confianza = float(v_probabilidades[v_idx].item() * 100)
            v_label_predicha = v_clases_dano[v_idx]

        # Consulta al tabulador con normalización robusta de datos y texto
        v_desc = "No esta dado de alta en el tabulador"
        v_monto = 0.0

        if v_df_tabulador is not None and not v_df_tabulador.empty:
            # Crear copia de trabajo para normalizar columnas y evitar espacios ocultos o mayúsculas
            v_df_eval = v_df_tabulador.copy()
            v_df_eval['marca_clean'] = v_df_eval['marca'].astype(str).str.strip().str.lower()
            v_df_eval['modelo_clean'] = v_df_eval['modelo'].astype(str).str.strip().str.lower()
            v_df_eval['tipo_clean'] = v_df_eval['tipo_vehiculo'].astype(str).str.strip().str.lower()
            v_df_eval['label_clean'] = v_df_eval['label_real'].astype(str).str.strip().str.lower()

            v_resultado = v_df_eval[
                (v_df_eval['marca_clean'] == str(marca).strip().lower()) &
                (v_df_eval['modelo_clean'] == str(modelo).strip().lower()) &
                (v_df_eval['anio'].astype(int) == int(anio)) &
                (v_df_eval['tipo_clean'] == str(tipo_vehiculo).strip().lower()) &
                (v_df_eval['label_clean'] == str(v_label_predicha).strip().lower())
            ]

            if not v_resultado.empty:
                v_desc = v_resultado.iloc[0]['descripcion_dano_es']
                v_monto = float(v_resultado.iloc[0]['monto_reparacion_estimado_mxn'])

        return {
            "vehiculo": {
                "marca": marca,
                "modelo": modelo,
                "anio": anio,
                "tipo": tipo_vehiculo
            },
            "evaluacion_ia": {
                "modelo_utilizado": "ViT-B/16",
                "label_tecnico": v_label_predicha,
                "descripcion_dano": v_desc,
                "certeza_porcentaje": round(v_confianza, 2)
            },
            "presupuesto": {
                "monto_estimado_reparacion_mxn": v_monto,
                "moneda": "MXN"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia o procesamiento: {str(e)}")