import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np

from ImageProcessor import ImageProcessor

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")

    # Basic styling
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

    # Initialize state variables
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
    
    # New Edge Detection section
    apply_edge_detection = st.sidebar.checkbox("Edge Detection", False)
    edge_threshold = st.sidebar.slider("Edge Threshold", 0, 255, 30, disabled=not apply_edge_detection)
    
    # Filters
    filter_type = st.sidebar.selectbox(
    "Filter", 
    ["None", "Average", "Gaussian", "Sharpen", "Custom"]
)
    custom_mask = None
    if filter_type == "Custom":
        st.sidebar.subheader("Custom Filter Matrix (3x3)")
        
        # Tworzenie siatki 3x3 do wprowadzania wag
        rows = []
        for i in range(3):
            cols = st.sidebar.columns(3)
            row_weights = []
            for j in range(3):
                with cols[j]:
                    weight = st.number_input(
                        f"Row {i+1}, Col {j+1}",
                        key=f"custom_filter_{i}_{j}",
                        value=0.0
                    )
                    row_weights.append(weight)
            rows.append(row_weights)
        
        custom_mask = np.array(rows, dtype=np.float32)
    
    
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

        ## reset when image changed
        if st.session_state.original_image != image_bytes:
            st.session_state.original_image = image_bytes
            
            temp_file = io.BytesIO(image_bytes)
            st.session_state.processor = ImageProcessor(temp_file)
            st.session_state.processed_image = None
            
            # Reset cached projections
            st.session_state.cached_hist_original = None
            st.session_state.cached_hist_processed = None
            st.session_state.cached_hproj_original = None
            st.session_state.cached_vproj_original = None
            st.session_state.cached_hproj_processed = None
            st.session_state.cached_vproj_processed = None
        
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
                
            # New edge detection
            if apply_edge_detection:
                processor.edge_detection(threshold=edge_threshold, processed=True)
                
            if filter_type != "None":
                if filter_type == "Custom":
                    if custom_mask is not None:
                        try:
                            processor.filter2(method="custom", custom_mask=custom_mask, processed=True)
                        except Exception as e:
                            st.error(f"Error applying custom filter: {str(e)}")
                    else:
                        st.error("Please provide a custom filter matrix!")
                else:
                    processor.filter2(method=filter_type.lower(), processed=True) 
            
            processed_img = Image.fromarray(processor.processed_pixels)
            buf = io.BytesIO()
            processed_img.save(buf, format="PNG")
            st.session_state.processed_image = buf.getvalue()
        
        # Original image display in first column
        with col1:
            st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)

            st.markdown("<h4 style='text-align: center;'>Original - Histogram</h4>", unsafe_allow_html=True)
            
            # Histogram
            if st.session_state.processor is not None:
                if st.session_state.cached_hist_original is None or reset_button:
                    plt.figure(figsize=(6, 3))
                    hist_fig_original = st.session_state.processor.make_histogram(processed=False)
                    st.session_state.cached_hist_original = hist_fig_original
                
                st.pyplot(st.session_state.cached_hist_original)
                
            # Projections for original image
            if show_horizontal_projection or show_vertical_projection:
                st.markdown("<h4 style='text-align: center;'>Original - Projections</h4>", unsafe_allow_html=True)
                # Create two sub-columns within the first main column
                proj_col1, proj_col2 = st.columns(2)
                
                # Horizontal projection in first sub-column
                if show_horizontal_projection:
                    with proj_col1:
                        st.markdown("<p style='text-align: center;'>Horizontal Projection</p>", unsafe_allow_html=True)
                        if st.session_state.processor is not None:
                            if st.session_state.cached_hproj_original is None or reset_button:
                                plt.figure(figsize=(4, 3))
                                _, hproj_fig_original = st.session_state.processor.horizontal_projection(processed=False)
                                st.session_state.cached_hproj_original = hproj_fig_original
                            
                            st.pyplot(st.session_state.cached_hproj_original)
                
                # Vertical projection in second sub-column
                if show_vertical_projection:
                    with proj_col2:
                        st.markdown("<p style='text-align: center;'>Vertical Projection</p>", unsafe_allow_html=True)
                        if st.session_state.processor is not None:
                            if st.session_state.cached_vproj_original is None or reset_button:
                                plt.figure(figsize=(4, 3))
                                _, vproj_fig_original = st.session_state.processor.vertical_projection(processed=False)
                                st.session_state.cached_vproj_original = vproj_fig_original
                            
                            st.pyplot(st.session_state.cached_vproj_original)
        
        # Processed image display in second column
        with col2:
            st.markdown("<h3 style='text-align: center;'>Processed Image</h3>", unsafe_allow_html=True)
            if st.session_state.processed_image is not None:
                st.image(st.session_state.processed_image, use_container_width=True)

                st.markdown("<h4 style='text-align: center;'>Processed - Histogram</h4>", unsafe_allow_html=True)
                
                # Histogram 
                if 'cached_hist_processed' not in st.session_state or apply_button:
                    plt.figure(figsize=(6, 3))
                    hist_fig_processed = st.session_state.processor.make_histogram(processed=True)
                    st.session_state.cached_hist_processed = hist_fig_processed
                
                st.pyplot(st.session_state.cached_hist_processed)
                
                # Projections for processed image
                if show_horizontal_projection or show_vertical_projection:
                    st.markdown("<h4 style='text-align: center;'>Processed - Projections</h4>", unsafe_allow_html=True)
                    # two sub-columns within the second main column
                    proj_col1, proj_col2 = st.columns(2)
                    
                    # Horizontal projection in first sub-column
                    if show_horizontal_projection:
                        with proj_col1:
                            st.markdown("<p style='text-align: center;'>Horizontal Projection</p>", unsafe_allow_html=True)
                            if st.session_state.cached_hproj_processed is None or apply_button:
                                plt.figure(figsize=(4, 3))
                                _, hproj_fig_processed = st.session_state.processor.horizontal_projection(processed=True)
                                st.session_state.cached_hproj_processed = hproj_fig_processed
                            
                            st.pyplot(st.session_state.cached_hproj_processed)
                    
                    # Vertical projection in second sub-column
                    if show_vertical_projection:
                        with proj_col2:
                            st.markdown("<p style='text-align: center;'>Vertical Projection</p>", unsafe_allow_html=True)
                            if st.session_state.cached_vproj_processed is None or apply_button:
                                plt.figure(figsize=(4, 3))
                                _, vproj_fig_processed = st.session_state.processor.vertical_projection(processed=True)
                                st.session_state.cached_vproj_processed = vproj_fig_processed
                            
                            st.pyplot(st.session_state.cached_vproj_processed)
            else:
                # If no processing has been done yet, show the original
                st.image(st.session_state.original_image, use_container_width=True)
                st.pyplot(st.session_state.cached_hist_original)
                
                # Show the original projections if they're enabled
                if (show_horizontal_projection or show_vertical_projection) and st.session_state.processor is not None:
                    st.markdown("<h4 style='text-align: center;'>Original - Projections</h4>", unsafe_allow_html=True)
                    # Two sub-columns within the second main column
                    proj_col1, proj_col2 = st.columns(2)
                    
                    if show_horizontal_projection:
                        with proj_col1:
                            st.markdown("<p style='text-align: center;'>Horizontal Projection</p>", unsafe_allow_html=True)
                            st.pyplot(st.session_state.cached_hproj_original)
                    
                    if show_vertical_projection:
                        with proj_col2:
                            st.markdown("<p style='text-align: center;'>Vertical Projection</p>", unsafe_allow_html=True)
                            st.pyplot(st.session_state.cached_vproj_original)
        
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