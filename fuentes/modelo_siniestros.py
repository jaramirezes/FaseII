import os
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.optim import AdamW
import pandas as pd
from collections import Counter

def cargar_y_entrenar_siniestros(directorio_modelo="./modelo_vit_siniestros"):
    print("Iniciando Carga y Procesamiento del Sistema")

    # Definición de rutas relativas locales para compatibilidad con Docker y GitHub
    v_directorio_actual = os.path.dirname(os.path.abspath(__file__))
    v_ruta_zip = os.path.join(v_directorio_actual, "carga", "imagenes_entrenamiento.zip")
    v_ruta_tabulador = os.path.join(v_directorio_actual, "carga", "tabulador_costos_siniestros.csv")

    print(f"\nArchivo ZIP identificado: {v_ruta_zip}")
    print(f"Archivo de tabulador identificado: {v_ruta_tabulador}")

    if os.path.exists(v_ruta_tabulador):
        try:
            if v_ruta_tabulador.endswith('.csv'):
                v_df_tabulador = pd.read_csv(v_ruta_tabulador)
            else:
                v_df_tabulador = pd.read_excel(v_ruta_tabulador)
            print(f"Tabulador de costos cargado exitosamente. Dimensiones: {v_df_tabulador.shape}")
        except Exception as e:
            print(f"Aviso al leer el tabulador: {e}")
    else:
        print(f"Aviso: No se encontró el tabulador en la ruta especificada.")

    v_directorio_extraccion = "./dataset_vehiculos"
    print("\nDescomprimir el dataset")
    if os.path.exists(v_ruta_zip):
        with zipfile.ZipFile(v_ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(v_directorio_extraccion)
        print(f"Dataset descomprimido exitosamente en: {v_directorio_extraccion}")
    else:
        print(f"Aviso: No se encontró el archivo ZIP en {v_ruta_zip}")

    # Extracción de clases de los XML
    print("\nObteniendo clases dinamicas")
    v_clases_set = set()

    if os.path.exists(v_directorio_extraccion):
        for root, dirs, files_in_dir in os.walk(v_directorio_extraccion):
            for file in files_in_dir:
                if file.endswith('.xml'):
                    v_xml_path = os.path.join(root, file)
                    try:
                        v_tree = ET.parse(v_xml_path)
                        v_root_xml = v_tree.getroot()
                        for v_obj in v_root_xml.findall('object'):
                            v_nombre_clase_elem = v_obj.find('name')
                            if v_nombre_clase_elem is not None and v_nombre_clase_elem.text:
                                v_clases_set.add(v_nombre_clase_elem.text.strip())
                    except Exception as e:
                        pass

    # Convertimos el conjunto a una lista ordenada para mantener índices estables
    v_clases_dano = sorted(list(v_clases_set))
    print(f"Clases detectadas ({len(v_clases_dano)}): {v_clases_dano}")

    if not os.path.exists(directorio_modelo) and len(v_clases_dano) > 0:
        os.makedirs(directorio_modelo, exist_ok=True)
        v_base_model = "google/vit-base-patch16-224"
        v_temp_processor = AutoImageProcessor.from_pretrained(v_base_model)
        
        # Crear los diccionarios de mapeo con los nombres reales de las clases dinámicas
        v_id2label = {v_i: v_c for v_i, v_c in enumerate(v_clases_dano)}
        v_label2id = {v_c: v_i for v_i, v_c in enumerate(v_clases_dano)}

        v_temp_model = AutoModelForImageClassification.from_pretrained(
            v_base_model, 
            num_labels=len(v_clases_dano), 
            id2label=v_id2label,
            label2id=v_label2id,
            ignore_mismatched_sizes=True
        )
        v_temp_processor.save_pretrained(directorio_modelo)
        v_temp_model.save_pretrained(directorio_modelo)

    class DatasetSiniestros(Dataset):
        def __init__(self, directorio_raiz, image_processor, clases_disponibles):
            self.directorio_raiz = directorio_raiz
            self.image_processor = image_processor
            self.clases_disponibles = clases_disponibles
            self.muestras = []
            v_conteo_clases = []

            if os.path.exists(directorio_raiz):
                for root, dirs, files_in_dir in os.walk(directorio_raiz):
                    for file in files_in_dir:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(root, file)
                            xml_path = os.path.splitext(img_path)[0] + ".xml"
                            clase_idx = 0 # Valor por defecto seguro

                            if os.path.exists(xml_path):
                                try:
                                    tree = ET.parse(xml_path)
                                    root_xml = tree.getroot()
                                    obj = root_xml.find('object')
                                    if obj is not None:
                                        nombre_clase_elem = obj.find('name')
                                        if nombre_clase_elem is not None and nombre_clase_elem.text:
                                            txt_clase = nombre_clase_elem.text.strip()
                                            if txt_clase in self.clases_disponibles:
                                                clase_idx = self.clases_disponibles.index(txt_clase)
                                except Exception as e:
                                    pass

                            self.muestras.append((img_path, clase_idx))
                            v_conteo_clases.append(clase_idx)

            print("\nDistribución de clases detectadas en XML")
            v_distribucion = Counter(v_conteo_clases)
            for idx, count in v_distribucion.items():
                print(f"   [{idx}] {self.clases_disponibles[idx]}: {count} imágenes")

        def __len__(self):
            return len(self.muestras)

        def __getitem__(self, idx):
            img_path, label = self.muestras[idx]
            image = Image.open(img_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, torch.tensor(label, dtype=torch.long)

    if os.path.exists(directorio_modelo):
        v_image_processor = AutoImageProcessor.from_pretrained(directorio_modelo, local_files_only=True)
    else:
        v_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    v_dataset = DatasetSiniestros(v_directorio_extraccion, v_image_processor, v_clases_dano)

    if len(v_dataset) > 0:
        v_dataloader = DataLoader(v_dataset, batch_size=4, shuffle=True)
        print(f"\nTotal de imágenes listas para entrenamiento: {len(v_dataset)}")

        v_model = AutoModelForImageClassification.from_pretrained(
            directorio_modelo, num_labels=len(v_clases_dano), ignore_mismatched_sizes=True, local_files_only=True
        )

        v_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Entrenando en dispositivo: {v_device}")
        v_model.to(v_device)
        v_model.train()

        v_optimizer = AdamW(v_model.parameters(), lr=2e-5)

        print("\nIniciando ciclo de entrenamiento corregido (5 épocas)")
        v_epocas = 5
        for epoca in range(v_epocas):
            v_suma_loss = 0.0
            for pixel_values, labels in v_dataloader:
                pixel_values, labels = pixel_values.to(v_device), labels.to(v_device)
                v_optimizer.zero_grad()
                v_outputs = v_model(pixel_values=pixel_values, labels=labels)
                v_loss = v_outputs.loss
                v_loss.backward()
                v_optimizer.step()
                v_suma_loss += v_loss.item()

            print(f"Época {epoca+1}/{v_epocas} completada - Loss promedio: {v_suma_loss / len(v_dataloader):.4f}")

        v_model.save_pretrained(directorio_modelo)
        v_image_processor.save_pretrained(directorio_modelo)
        print(f"\nEntrenamiento finalizado, Modelo guardado en: {directorio_modelo}")
    else:
        print("No se encontraron imágenes válidas para entrenar.")

# Este bloque simula la ejecución que hacías en la otra celda de Colab
if __name__ == "__main__":
    cargar_y_entrenar_siniestros()