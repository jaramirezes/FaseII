import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

def ejecuta_simulacion():
    print("Iniciando Carga y Procesamiento del Sistema")

    # Rutas relativas adaptadas para entorno local en VS Code
    v_directorio_actual = os.path.dirname(os.path.abspath(__file__))
    v_ruta_tabulador = os.path.join(v_directorio_actual, "carga", "tabulador_costos_siniestros.csv")

    # 1. Cargar el tabulador de costos
    v_df_tabulador = pd.read_csv(v_ruta_tabulador)

    # 2. Cargar el modelo entrenado y su procesador desde la carpeta guardada
    v_directorio_modelo = os.path.join(v_directorio_actual, "modelo_vit_siniestros")
    print("Cargando el modelo ViT y el procesador de imágenes...")
    v_image_processor = AutoImageProcessor.from_pretrained(v_directorio_modelo, local_files_only=True)
    v_modelo_vit = AutoModelForImageClassification.from_pretrained(v_directorio_modelo, local_files_only=True)

    # 3. Extraer las clases de daño desde la configuración del modelo entrenado
    v_id2label = v_modelo_vit.config.id2label
    v_clases_dano = [v_id2label[v_i] for v_i in sorted(v_id2label.keys())]
    print(f"Clases cargadas dinámicamente del modelo ({len(v_clases_dano)}): {v_clases_dano}")

    v_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_modelo_vit.to(v_device)
    v_modelo_vit.eval()


    # 4. Función que manda llamar el modelo con la imagen
    def predecir_con_modelo_real(v_nombre_imagen):
        try:
            if not v_nombre_imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
                v_nombre_imagen += ".jpg"

            v_ruta_final = None
            v_directorio_dataset = os.path.join(v_directorio_actual, "dataset_vehiculos")
            for v_root, v_dirs, v_files in os.walk(v_directorio_dataset):
                if v_nombre_imagen in v_files:
                    v_ruta_final = os.path.join(v_root, v_nombre_imagen)
                    break

            if not v_ruta_final or not os.path.exists(v_ruta_final):
                v_ruta_final = v_nombre_imagen

            v_image = Image.open(v_ruta_final).convert("RGB")
            v_inputs = v_image_processor(images=v_image, return_tensors="pt")
            v_pixel_values = v_inputs["pixel_values"].to(v_device)

            with torch.no_grad():
                v_outputs = v_modelo_vit(pixel_values=v_pixel_values)
                v_logits = v_outputs.logits
                v_probabilidades = torch.nn.functional.softmax(v_logits, dim=-1)[0]
                v_predicted_class_idx = v_logits.argmax(-1).item()
                v_confianza = v_probabilidades[v_predicted_class_idx].item() * 100
            return v_clases_dano[v_predicted_class_idx], v_confianza  
              
        except Exception as v_e:
            print(f"Error procesando la imagen {v_nombre_imagen}: {v_e}")
            return "Desconocido"

    # 5. Transacciones de casos de prueba.
    v_transacciones = [
        {"marca": "Volkswagen", "modelo": "Tiguan", "anio": 2018, "tipo_vehiculo": "SUV", "imagen": "25_jpeg.rf.7d6a010838584cb14aaa1d883d1e8a86"},
        {"marca": "Ford", "modelo": "Escape", "anio": 2026, "tipo_vehiculo": "SUV", "imagen": "back_light-1---1-_jpg.rf.115a73704cb5284543451ddd7f68ffb2"},
        {"marca": "Chevrolet", "modelo": "Tracker", "anio": 2023, "tipo_vehiculo": "SUV", "imagen": "sign_light--26-_jpg.rf.359e8219cd2b69dce60e9fdad8db34e2.jpg"},
        {"marca": "Nissan", "modelo": "Tsuru", "anio": 2010, "tipo_vehiculo": "SUV", "imagen": "1521512941709--1-_jpg.rf.3acf1e6dd65be1cb4fe88d56759024c5.jpg"},
        {"marca": "Nissan", "modelo": "Kicks", "anio": 2018, "tipo_vehiculo": "SUV", "imagen": "car_fender--23-_jpg.rf.891c897157dbe95c80e65ff0162f6cca.jpg"}
    ]

    print("\nPrueba de evaluación de siniestros con modelo VIT")

    for v_i, v_t in enumerate(v_transacciones):
        v_dano_calculado_por_modelo, v_confianza = predecir_con_modelo_real(v_t["imagen"])

        # Consulta al tabulador
        v_resultado = v_df_tabulador[
            (v_df_tabulador['marca'].str.lower() == v_t['marca'].lower()) &
            (v_df_tabulador['modelo'].str.lower() == v_t['modelo'].lower()) &
            (v_df_tabulador['anio'] == int(v_t['anio'])) &
            (v_df_tabulador['tipo_vehiculo'].str.lower() == v_t['tipo_vehiculo'].lower()) &
            (v_df_tabulador['label_real'].str.lower() == v_dano_calculado_por_modelo.lower())
        ]


        if not v_resultado.empty:
            v_desc = v_resultado.iloc[0]['descripcion_dano_es']
            v_monto = v_resultado.iloc[0]['monto_reparacion_estimado_mxn']
        else:
            v_desc = "No fue encontrado en el tabulador, requiere valuación manual."
            v_monto = 0.0
        print(f"Transacción {v_i+1}:")
        print(f"  Vehículo: {v_t['marca']} {v_t['modelo']} {v_t['anio']} {v_t['tipo_vehiculo']}  Daño calculado: {v_dano_calculado_por_modelo} (Confianza: {v_confianza:.2f}%)")
        print(f"  Descripción            : {v_desc}")
        print(f"  Importe del daño       : ${v_monto:,.2f} MXN\n")

if __name__ == "__main__":
    ejecuta_simulacion()