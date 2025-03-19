import streamlit as st
from PIL import Image
import io

from ImageProcessor import ImageProcessor

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")
    
    st.title("Image Processing App")
    
    # Initialize session state for storing image processor
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Sidebar with options
    st.sidebar.title("Processing Options")
    st.sidebar.subheader("Basic Operations")
    
    apply_grayscale = st.sidebar.checkbox("Grayscale", False)
    grayscale_method = st.sidebar.selectbox(
        "Grayscale Method", 
        ["Luminosity", "Average", "Luminance", "Desaturation", "Decomposition-max", "Decomposition-min"],
        disabled=not apply_grayscale
    )
    
    apply_negative = st.sidebar.checkbox("Negative", False)
    
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
    
    apply_binarize = st.sidebar.checkbox("Binarize", False)
    threshold = st.sidebar.slider("Threshold", 0, 255, 128, disabled=not apply_binarize)
    
    # Filters
    filter_type = st.sidebar.selectbox(
        "Filter", 
        ["None", "Average", "Gaussian", "Sharpen"]
    )
    
    # Reset and Apply buttons next to each other
    col1, col2 = st.sidebar.columns(2)
    with col1:
        reset_button = st.button("Reset", use_container_width=True)
    with col2:
        apply_button = st.button("Apply", use_container_width=True)
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Two columns for displaying original and processed images side by side
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read the uploaded image
        image_bytes = uploaded_file.getvalue()
        
        # Save the original image to session state
        if st.session_state.original_image is None or reset_button:
            st.session_state.original_image = image_bytes
            
            temp_file = io.BytesIO(image_bytes)
            st.session_state.processor = ImageProcessor(temp_file)
            st.session_state.processed_image = None
        
        # Original image display
        with col1:
            st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
        
        # Process the image when Apply button is clicked
        if apply_button and st.session_state.processor is not None:
            processor = st.session_state.processor
            # Reset to the original image
            processor.reset()
            
            # Apply selected operations
            if apply_grayscale:
                processor.grayscale(method=grayscale_method.lower(), processed=True)
            
            if brightness != 0:
                processor.adjust_brightness(brightness, processed=True)
                
            if contrast != 0:
                processor.contrast_correction(contrast, processed=True)
                
            if apply_negative:
                processor.negative(processed=True)
                
            if apply_binarize:
                processor.binarize(threshold, processed=True)
                
            if filter_type != "None":
                processor.filter2(method=filter_type.lower(), processed=True)
            
            processed_img = Image.fromarray(processor.processed_pixels)
            buf = io.BytesIO()
            processed_img.save(buf, format="PNG")
            st.session_state.processed_image = buf.getvalue()
        
        # Processed image display
        with col2:
            st.markdown("<h3 style='text-align: center;'>Processed Image</h3>", unsafe_allow_html=True)
            if st.session_state.processed_image is not None:
                st.image(st.session_state.processed_image, use_container_width=True)
            else:
                # If no processing has been done yet, show the original
                st.image(st.session_state.original_image, use_container_width=True)
        
        # Download button appears when processed image is available
        if st.session_state.processed_image is not None:
            st.sidebar.download_button(
                label="Download Processed Image",
                data=st.session_state.processed_image,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()