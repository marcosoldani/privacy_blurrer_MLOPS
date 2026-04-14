import streamlit as st
import requests
from PIL import Image
import io

# 1. Configurazione base della pagina web
st.set_page_config(page_title="Privacy Blurrer Frontend", layout="centered")
st.title("Privacy Blurrer 🕵️‍♂️")
st.write("Carica un'immagine per generare la maschera di segmentazione delle persone.")

# Indirizzo dove il tuo backend FastAPI è in ascolto
API_URL = "http://localhost:8000/predict"

# 2. Creiamo il componente per far caricare l'immagine all'utente
uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine Originale", use_container_width=True)
    
    # Creiamo un pulsante che, se premuto, avvia l'elaborazione
    if st.button("Genera Maschera"):
        
        # Mostra un'icona di caricamento mentre aspettiamo il backend
        with st.spinner("Elaborazione in corso col modello U-Net..."):
            
            # Prepariamo il file da inviare tramite richiesta HTTP
            # Riposizioniamo il puntatore di lettura all'inizio del file
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # 3. Inviamo l'immagine al backend usando una richiesta POST
                response = requests.post(API_URL, files=files)
                
                # Se il backend risponde con "200 OK", tutto è andato bene
                if response.status_code == 200:
                    
                    # 4. Leggiamo l'immagine restituita dal backend
                    mask_bytes = response.content
                    mask_image = Image.open(io.BytesIO(mask_bytes))
                    
                    # Mostriamo la maschera a schermo!
                    st.image(mask_image, caption="Maschera Generata (Output del Modello)", use_container_width=True)
                    st.success("Segmentazione completata con successo!")
                    
                else:
                    # Se c'è un errore dal backend (es. formato sbagliato)
                    st.error(f"Errore dal server FastAPI: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                # Se il frontend non riesce a trovare il backend
                st.error("Impossibile connettersi al backend!")
                st.info("Assicurati di aver avviato FastAPI con: uvicorn src.app:app --host 0.0.0.0 --port 8000")