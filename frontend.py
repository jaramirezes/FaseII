import streamlit as st
import requests

st.title("🚗 Evaluación de Siniestros Automotrices")
st.write("Sube una imagen y llena los datos para evaluar el siniestro.")

# Formulario
with st.form("mi_formulario"):
    marca = st.text_input("Marca", "Volkswagen")
    modelo = st.text_input("Modelo", "Tiguan")
    anio = st.number_input("Año", 2010, 2025, 2018)
    tipo = st.selectbox("Tipo de Vehículo", ["SUV", "Sedán", "Hatchback", "Pickup"])
    imagen = st.file_uploader("Sube la imagen del siniestro", type=["jpg", "jpeg", "png"])
    
    submit = st.form_submit_button("Evaluar Siniestro")

if submit:
    if imagen is not None:
        # Preparamos los datos para enviar a FastAPI
        files = {"imagen": (imagen.name, imagen.getvalue(), imagen.type)}
        data = {"marca": marca, "modelo": modelo, "anio": anio, "tipo_vehiculo": tipo}
        
        # Hacemos la petición a tu backend
        try:
            response = requests.post("http://127.0.0.1:8000/predict", data=data, files=files)
            
            if response.status_code == 200:
                resultado = response.json()
                st.success("¡Evaluación exitosa!")
                st.json(resultado) # Muestra el JSON de respuesta bonito
            else:
                st.error(f"Error en la API: {response.status_code}")
        except Exception as e:
            st.error(f"No se pudo conectar con el backend: {e}")
    else:
        st.warning("Por favor, sube una imagen.")