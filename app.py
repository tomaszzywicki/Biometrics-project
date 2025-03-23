import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt

from ImageProcessor import ImageProcessor

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")

    # to jakieś do stylów ale w sumie to nadal się rozjeżdza
    st.markdown("""
        <style>
        .column-center {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # Żeby się nie ruszał ten og histogram potem
    if 'cached_hist_original' not in st.session_state:
        st.session_state.cached_hist_original = None
    
    # Cache for projections
    if 'cached_hproj_original' not in st.session_state:
        st.session_state.cached_hproj_original = None
    if 'cached_vproj_original' not in st.session_state:
        st.session_state.cached_vproj_original = None
    if 'cached_hproj_processed' not in st.session_state:
        st.session_state.cached_hproj_processed = None
    if 'cached_vproj_processed' not in st.session_state:
        st.session_state.cached_vproj_processed = None
    
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
    
    st.sidebar.subheader("Display Options")
    show_horizontal_projection = st.sidebar.checkbox("Show Horizontal Projection", False)
    show_vertical_projection = st.sidebar.checkbox("Show Vertical Projection", False)
    
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
            
            # Reset cached projections
            st.session_state.cached_hproj_original = None
            st.session_state.cached_vproj_original = None
            st.session_state.cached_hproj_processed = None
            st.session_state.cached_vproj_processed = None
        
        # Original image display
        with col1:
            st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
            # Create nested columns for centering
            img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
            with img_col2:
                st.image(st.session_state.original_image, width=400)
        
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
            # Create nested columns for centering
            img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
            with img_col2:
                if st.session_state.processed_image is not None:
                    st.image(st.session_state.processed_image, width=400)
                else:
                    # If no processing has been done yet, show the original
                    st.image(st.session_state.original_image, width=400)
        
        # Download button appears when processed image is available
        if st.session_state.processed_image is not None:
            st.sidebar.download_button(
                label="Download Processed Image",
                data=st.session_state.processed_image,
                file_name="processed_image.png",
                mime="image/png"
            )

        # Histogramy
        hist_col1, hist_col2 = st.columns(2)

        # Histogram oryginalnego obrazu - generowany tylko raz przy załadowaniu
        with hist_col1:
            if st.session_state.processor is not None:
                if st.session_state.cached_hist_original is None or reset_button:
                    # Set figure size explicitly here if make_histogram doesn't do it
                    plt.figure(figsize=(4, 3))
                    hist_fig_original = st.session_state.processor.make_histogram(processed=False)
                    st.session_state.cached_hist_original = hist_fig_original
                
                # Center align with a fixed width container
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(st.session_state.cached_hist_original, use_container_width=False)

        # Histogram przetworzonego obrazu - aktualizowany tylko po kliknięciu Apply
        with hist_col2:
            if st.session_state.processed_image is not None:
                # Użyj "cached_hist_processed" do przechowywania histogramu
                if 'cached_hist_processed' not in st.session_state or apply_button:
                    # Aktualizuj histogram tylko gdy naciśnięto Apply
                    plt.figure(figsize=(4, 3))
                    hist_fig_processed = st.session_state.processor.make_histogram(processed=True)
                    st.session_state.cached_hist_processed = hist_fig_processed
                
                # Center align with a fixed width container
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(st.session_state.cached_hist_processed, use_container_width=False)
            else:
                # Center align with a fixed width container
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(st.session_state.cached_hist_original, use_container_width=False)
        
        # Projekcja pozioma
        if show_horizontal_projection:
            hproj_col1, hproj_col2 = st.columns(2)
            
            with hproj_col1:
                if st.session_state.processor is not None:
                    # Generate projections only once or on reset
                    if st.session_state.cached_hproj_original is None or reset_button:
                        plt.figure(figsize=(4, 3))
                        _, hproj_fig_original = st.session_state.processor.horizontal_projection(processed=False)
                        st.session_state.cached_hproj_original = hproj_fig_original
                    
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_hproj_original, use_container_width=False)
            
            with hproj_col2:
                if st.session_state.processed_image is not None:
                    # Update projections only when Apply is clicked
                    if st.session_state.cached_hproj_processed is None or apply_button:
                        plt.figure(figsize=(4, 3))
                        _, hproj_fig_processed = st.session_state.processor.horizontal_projection(processed=True)
                        st.session_state.cached_hproj_processed = hproj_fig_processed
                    
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_hproj_processed, use_container_width=False)
                else:
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_hproj_original, use_container_width=False)
        
        # Projekcja pionowa
        if show_vertical_projection:
            vproj_col1, vproj_col2 = st.columns(2)
            
            with vproj_col1:
                if st.session_state.processor is not None:
                    # Generate projections only once or on reset
                    if st.session_state.cached_vproj_original is None or reset_button:
                        plt.figure(figsize=(4, 3))
                        _, vproj_fig_original = st.session_state.processor.vertical_projection(processed=False)
                        st.session_state.cached_vproj_original = vproj_fig_original
                    
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_vproj_original, use_container_width=False)
            
            with vproj_col2:
                if st.session_state.processed_image is not None:
                    # Update projections only when Apply is clicked
                    if st.session_state.cached_vproj_processed is None or apply_button:
                        plt.figure(figsize=(4, 3))
                        _, vproj_fig_processed = st.session_state.processor.vertical_projection(processed=True)
                        st.session_state.cached_vproj_processed = vproj_fig_processed
                    
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_vproj_processed, use_container_width=False)
                else:
                    # Center align with a fixed width container
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.pyplot(st.session_state.cached_vproj_original, use_container_width=False)

if __name__ == "__main__":
    main()