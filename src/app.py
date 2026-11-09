import streamlit as st
from PIL import Image
import torch
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(
    page_title=" Demo Progetto SPAI",
    layout="wide"
)

st.title("🔬 Tiny Object Detection per Spermatozoi")
st.markdown("Interfaccia di inferenza e valutazione per il confronto dei modelli e dei regimi di training")

# SIDEBAR 
st.sidebar.header("Configurazione Demo")
model_choice = st.sidebar.selectbox("Seleziona Architettura", ["yolov8", "rtdetr"])
regime_choice = st.sidebar.selectbox("Regime di Training", ["regime_b", "regime_a"])

weights_path = Path(f"runs/detect/runs/train/{model_choice}_{regime_choice}/weights/best.pt")

st.sidebar.markdown("---")
st.sidebar.subheader("Stato del Modello")
if weights_path.exists():
    st.sidebar.success(f"Pesi trovati: {weights_path.name}")
    model_to_load = str(weights_path)
else:
    st.sidebar.warning("Pesi non ancora pronti (training in corso). Uso il modello base per la demo")
    model_to_load = "yolov8n.pt"

# Caricamento del modello
@st.cache_resource
def load_detection_model(path):
    return YOLO(path)

with st.spinner("Caricamento del modello in corso..."):
    net = load_detection_model(model_to_load)

# CORPO
st.markdown("Carica un'immagine microscopica dal Test Set (SEM-0002 / SEM-0013)")

uploaded_file = st.file_uploader("Scegli un'immagine (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Immagine Originale")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Risultato Rilevamento")
        with st.spinner("Esecuzione inferenza a 1280px..."):
            results = net(image, imgsz=1280, conf=0.25)
            res_image = results[0].plot() # Rappresentazione dei bounding box

            # Convertimento da BGR a RGB
            res_image_rgb = Image.fromarray(res_image[..., ::-1])
            st.image(res_image_rgb, use_container_width=True)
            
        st.success("Inferenza completata con successo")
        
        # Metriche rilevate nel box
        boxes = results[0].boxes
        st.info(f"**Spermatozoi rilevati:** {len(boxes)}")
else:
    st.info("Carica un'immagine di prova per avviare il rilevamento dei tiny objects")