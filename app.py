import streamlit as st
from PIL import Image

from ImageProcessor import ImageProcessor

def main():
    st.set_page_config(page_title="Przetwarzanie obrazów", layout="wide")
    
    st.title("Image Processing App")
    
    # Sidebar z opcjami
    st.sidebar.title("Processing Options")
    st.sidebar.subheader("Podstawowe operacje")
    st.sidebar.checkbox("Greyscale")
    st.sidebar.checkbox("Negative")
    # Jasność
    st.sidebar.slider("Brightness", -100, 100, 0)
    # Kontrast
    st.sidebar.slider("Contrast", -100, 100, 0) 
    # Rozmycie
    st.sidebar.slider("Blur", 0, 10, 0)
    # Ostrzenie
    st.sidebar.slider("Sharpen", 0, 10, 0)
    
    # Uploadowanie obrazu
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        

if __name__ == "__main__":
    main()