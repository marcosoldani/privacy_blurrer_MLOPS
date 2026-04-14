import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Privacy Blurrer", layout="centered")
st.title("Privacy Blurrer 🕵️‍♂️")
st.write("Carica un'immagine per anonimizzare le persone rilevate dal modello U-Net.")

BASE_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine Originale", use_container_width=True)

    st.divider()

    # ── Quattro bottoni affiancati ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    action = None

    with col1:
        if st.button("🎭 Maschera", use_container_width=True):
            action = "predict"
    with col2:
        if st.button("🌫️ Gaussian", use_container_width=True):
            action = "gaussian"
    with col3:
        if st.button("🟦 Pixelate", use_container_width=True):
            action = "pixelate"
    with col4:
        if st.button("⬛ Blackout", use_container_width=True):
            action = "blackout"

    # ── Esecuzione e risultato grande sotto ───────────────────────────────────
    if action is not None:
        label_map = {
            "predict":  "Maschera binaria",
            "gaussian": "Gaussian Blur",
            "pixelate": "Pixelate",
            "blackout": "Blackout",
        }
        spinner_map = {
            "predict":  "Generazione maschera con U-Net...",
            "gaussian": "Applicazione Gaussian blur...",
            "pixelate": "Applicazione pixelate...",
            "blackout": "Applicazione blackout...",
        }

        with st.spinner(spinner_map[action]):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                if action == "predict":
                    response = requests.post(f"{BASE_URL}/predict", files=files)
                else:
                    response = requests.post(f"{BASE_URL}/blur?blur_type={action}", files=files)

                if response.status_code == 200:
                    result_image = Image.open(io.BytesIO(response.content))
                    st.image(result_image, caption=label_map[action], use_container_width=True)
                    st.success("Elaborazione completata!")
                else:
                    st.error(f"Errore dal server ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Backend non raggiungibile.")
                st.info("Avvia FastAPI con: uvicorn src.app:app --host 0.0.0.0 --port 8000")