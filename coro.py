import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import itertools

# Page configuration
st.set_page_config(page_title="Coronary Stenosis Detection and Segmentation App", layout="wide")

# Indicator that options are in the sidebar
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Options are available in the sidebar.</p>",
    unsafe_allow_html=True
)

# Load YOLO models once at the start
@st.cache_resource
def load_models():
    model_stenosis_path = "model.pt"  # Stenosis detection model
    model_segments_path = "model_s.pt"  # Segments identification model
    if not os.path.exists(model_stenosis_path):
        st.error(f"Stenosis model not found at the specified path: {model_stenosis_path}")
        st.stop()
    if not os.path.exists(model_segments_path):
        st.error(f"Segments model not found at the specified path: {model_segments_path}")
        st.stop()
    model_stenosis = YOLO(model_stenosis_path)
    model_segments = YOLO(model_segments_path)
    return model_stenosis, model_segments

model_stenosis, model_segments = load_models()

# Sidebar for controls
st.sidebar.header("Configuration")

# Slider for confidence threshold
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Slider for stenosis mask transparency
transparency_stenosis = st.sidebar.slider("Stenosis Mask Transparency", 0.0, 1.0, 0.25, 0.05)

# Slider for segment contours transparency
transparency_segments = st.sidebar.slider("Segment Contours Transparency", 0.0, 1.0, 0.3, 0.05)

# Function to load test images
def load_test_images():
    test_images = {}
    test_image_names = ["1.png", "2.png", "3.png"]  # Ensure these are PNG files
    for img_name in test_image_names:
        if os.path.exists(img_name):
            image = Image.open(img_name).convert("RGB")
            test_images[img_name] = image
        else:
            st.warning(f"Test image not found: {img_name}")
    return test_images

# Callback to select a test image
def select_test_image(img_name):
    st.session_state['selected_test_image'] = img_name

# Initialize selected image state
if 'selected_test_image' not in st.session_state:
    st.session_state['selected_test_image'] = None

# Main section: Upload an image or select a test image

# Option to upload an image
uploaded_file = st.file_uploader("Upload Your Own Image", type=["png", "jpg", "jpeg"])

# Variable to store the image to process
image_to_process = None

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_to_process = image
elif st.session_state['selected_test_image'] is not None:
    # Load the selected test image
    selected_image = Image.open(st.session_state['selected_test_image']).convert("RGB")
    image_np = np.array(selected_image)
    image_to_process = selected_image

# Show test images only if no image is uploaded or selected
if uploaded_file is None and st.session_state['selected_test_image'] is None:
    test_images = load_test_images()
    if test_images:
        st.subheader("Or Choose a Test Image:")
        cols = st.columns(3)
        for idx, (img_name, img) in enumerate(test_images.items()):
            with cols[idx]:
                # Create a thumbnail of the image
                thumbnail = img.copy()
                thumbnail.thumbnail((int(img.width * 0.3), int(img.height * 0.3)))  # 30% of original size
                # Display the thumbnail
                st.image(thumbnail, use_column_width=False, clamp=True)
                # Make the image clickable using a button below
                # Assign a unique key to each button to avoid duplication
                st.button(f"Select {img_name}", key=f"select_test_{img_name}", on_click=select_test_image, args=(img_name,))
else:
    # If an image is uploaded or selected, hide test images
    pass

# Definition of segment names
segment_number_to_name = {
    '1': 'RCA proximal',
    '2': 'RCA mid',
    '3': 'RCA distal',
    '4': 'Posterior descending',
    '5': 'Left main',
    '6': 'LAD proximal',
    '7': 'LAD mid',
    '8': 'LAD apical',
    '9': 'First diagonal',
    '9a': 'First diagonal a',
    '10': 'Second diagonal',
    '10a': 'Second diagonal a',
    '11': 'Proximal circumflex',
    '12': 'Intermediate/anterolateral',
    '12a': 'Obtuse marginal a',
    '12b': 'Obtuse marginal b',
    '13': 'Distal circumflex',
    '14': 'Left posterolateral',
    '14a': 'Left posterolateral a',
    '14b': 'Left posterolateral b',
    '15': 'Posterior descending',
    '16': 'Posterolateral from RCA',
    '16a': 'Posterolateral from RCA a',
    '16b': 'Posterolateral from RCA b',
    '16c': 'Posterolateral from RCA c',
}

# Process the image if available
if image_to_process is not None:
    try:
        # Create two columns to display images side by side
        # First column: Image with Masks
        # Second column: Original Image
        col_masks, col_original = st.columns(2)

        with col_masks:
            st.subheader("Stenosis and Segment Masks")
            # Perform inference with both models
            with st.spinner('Processing...'):
                results_stenosis = model_stenosis.predict(source=image_np, conf=threshold, save=False, task='segment')
                results_segments = model_segments.predict(source=image_np, conf=threshold, save=False, task='segment')

            # Process results
            if results_stenosis and results_segments:
                result_stenosis = results_stenosis[0]  # Assuming only one image is processed
                result_segments = results_segments[0]

                if (result_stenosis.masks is not None and len(result_stenosis.masks.data) > 0 and
                    result_segments.masks is not None and len(result_segments.masks.data) > 0):

                    # Extract masks and classes
                    masks_stenosis = result_stenosis.masks.data.cpu().numpy()  # (num_masks, height, width)
                    confidences_stenosis = result_stenosis.boxes.conf.cpu().numpy()  # Confidences
                    classes_stenosis = result_stenosis.boxes.cls.cpu().numpy().astype(int)
                    names_stenosis = model_stenosis.names  # Class names

                    masks_segments = result_segments.masks.data.cpu().numpy()
                    confidences_segments = result_segments.boxes.conf.cpu().numpy()
                    classes_segments = result_segments.boxes.cls.cpu().numpy().astype(int)
                    names_segments = model_segments.names

                    # Assign unique colors to each stenosis mask
                    color_palette_stenosis = [
                        (255, 0, 0),      # Red
                        (255, 165, 0),    # Orange
                        (255, 255, 0),    # Yellow
                        (255, 69, 0),     # Orange Red
                        (255, 215, 0),    # Gold
                        (255, 99, 71),    # Tomato
                        (255, 140, 0),    # Dark Orange
                        (255, 127, 80),    # Coral
                        (255, 105, 180),  # Hot Pink
                        (255, 20, 147),    # Deep Pink
                        (255, 0, 255),    # Magenta
                        (238, 130, 238),  # Violet
                    ]
                    color_cycle_stenosis = itertools.cycle(color_palette_stenosis)

                    mask_colors_stenosis = {}
                    for idx_mask in range(len(masks_stenosis)):
                        mask_colors_stenosis[idx_mask] = next(color_cycle_stenosis)

                    # Assign unique colors to each segment mask
                    color_palette_segments = [
                        (0, 0, 255),      # Blue
                        (0, 255, 0),      # Green
                        (0, 255, 255),    # Cyan
                        (138, 43, 226),   # Blue Violet
                        (75, 0, 130),     # Indigo
                        (0, 128, 128),    # Teal
                        (34, 139, 34),    # Forest Green
                        (255, 182, 193),  # Light Pink
                        (173, 216, 230),  # Light Blue
                        (0, 100, 0),      # Dark Green
                        (128, 0, 128),    # Purple
                        (255, 0, 0),      # Red (may overlap, adjust if necessary)
                    ]
                    color_cycle_segments = itertools.cycle(color_palette_segments)

                    mask_colors_segments = {}
                    for idx_mask in range(len(masks_segments)):
                        mask_colors_segments[idx_mask] = next(color_cycle_segments)

                    # Create a copy of the original image to overlay masks
                    img_with_masks = image_np.copy()

                    # List to store stenosis mask information
                    stenosis_info_list = []

                    # Preprocess all stenoses to get segmentation information
                    stenosis_overlaps = []

                    for idx_stenosis, (cls, conf) in enumerate(zip(classes_stenosis, confidences_stenosis)):
                        mask_stenosis = masks_stenosis[idx_stenosis]
                        mask_stenosis_resized = cv2.resize(mask_stenosis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # Determine overlapping segments
                        overlapping_segments = set()
                        for idx_segment, mask_segment in enumerate(masks_segments):
                            mask_segment_resized = cv2.resize(mask_segment, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                            # Calculate overlap
                            intersection = np.logical_and(mask_stenosis_resized > 0.5, mask_segment_resized > 0.5)
                            if np.any(intersection):
                                class_segment = classes_segments[idx_segment]
                                segment_number = names_segments[class_segment]
                                segment_name = segment_number_to_name.get(segment_number, 'Unknown')
                                overlapping_segments.add(segment_name)

                        stenosis_overlaps.append(overlapping_segments)

                    # Sidebar section for Localized Stenosis
                    st.sidebar.subheader("Localized Stenosis")

                    for idx_stenosis, (cls, conf) in enumerate(zip(classes_stenosis, confidences_stenosis)):
                        color_rgb = mask_colors_stenosis[idx_stenosis]
                        color_hex = '#%02x%02x%02x' % color_rgb
                        # Create a small color box using markdown
                        color_box = f'<div style="width: 15px; height: 15px; background-color: {color_hex}; display: inline-block; margin-right: 5px;"></div>'
                        overlapping_segments = stenosis_overlaps[idx_stenosis]
                        # Format overlapping segments
                        if overlapping_segments:
                            segments_str = ', '.join(overlapping_segments)
                        else:
                            segments_str = 'N/A'
                        # Format the label
                        label = f"{segments_str} (Confidence: {conf:.2f})"
                        # Create columns for color and checkbox
                        col_color, col_checkbox = st.sidebar.columns([1, 10])
                        with col_color:
                            st.markdown(color_box, unsafe_allow_html=True)
                        with col_checkbox:
                            # Use a unique key for each checkbox
                            mask_selection = st.checkbox(label, value=True, key=f"mask_checkbox_{idx_stenosis}")

                        if mask_selection:
                            # Process the stenosis mask
                            mask_stenosis = masks_stenosis[idx_stenosis]
                            mask_stenosis_resized = cv2.resize(mask_stenosis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                            # Overlay the stenosis mask on the image
                            color_mask = np.zeros_like(image_np)
                            color_mask[mask_stenosis_resized > 0.5] = color_rgb
                            img_with_masks = cv2.addWeighted(img_with_masks, 1.0, color_mask, transparency_stenosis, 0)

                            # Store the information
                            # Assuming each stenosis corresponds to at least one segment
                            stenosis_class = list(overlapping_segments)[0] if overlapping_segments else 'N/A'
                            stenosis_info = {
                                'stenosis_index': idx_stenosis,
                                'stenosis_class': stenosis_class,
                                'stenosis_confidence': conf,
                                'overlapping_segments_names': overlapping_segments,
                                'color_rgb': color_rgb,
                            }
                            stenosis_info_list.append(stenosis_info)

                    # Sidebar section for Identified Segments
                    st.sidebar.subheader("Identified Segments")

                    for idx_segment, (cls_seg, conf_seg) in enumerate(zip(classes_segments, confidences_segments)):
                        color_rgb_seg = mask_colors_segments[idx_segment]
                        color_hex_seg = '#%02x%02x%02x' % color_rgb_seg
                        # Create a small contour box using CSS
                        color_box_seg = f'''
                            <div style="
                                width: 15px; 
                                height: 15px; 
                                border: 2px solid {color_hex_seg}; 
                                background-color: transparent; 
                                display: inline-block; 
                                margin-right: 5px;">
                            </div>
                        '''
                        segment_number = names_segments[cls_seg]
                        segment_description = segment_number_to_name.get(segment_number, 'Unknown')
                        label_seg = f"◻️ {segment_description} (Confidence: {conf_seg:.2f})"
                        # Create columns for contour and checkbox
                        col_color_seg, col_checkbox_seg = st.sidebar.columns([1, 10])
                        with col_color_seg:
                            st.markdown(color_box_seg, unsafe_allow_html=True)
                        with col_checkbox_seg:
                            # Use a unique key for each checkbox and keep them unchecked by default
                            mask_selection_seg = st.checkbox(label_seg, value=False, key=f"mask_checkbox_seg_{idx_segment}")

                        if mask_selection_seg:
                            # Process the segment mask
                            mask_segment = masks_segments[idx_segment]
                            mask_segment_resized = cv2.resize(mask_segment, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                            # Find contours of the mask
                            mask_uint8 = (mask_segment_resized > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Draw finer contours on the image
                            cv2.drawContours(img_with_masks, contours, -1, color_rgb_seg, 1)  # Thickness = 1

                    # Display the image with masks
                    st.image(img_with_masks, use_column_width=True, clamp=True)

                    # Display stenosis and segments information
                    st.subheader("Stenosis and Segments Results")
                    for info in stenosis_info_list:
                        color_rgb = info['color_rgb']
                        color_hex = '#%02x%02x%02x' % color_rgb
                        color_box = f'<div style="width: 15px; height: 15px; background-color: {color_hex}; display: inline-block; margin-right: 5px;"></div>'
                        st.markdown(f"{color_box} **{info['stenosis_class']}**", unsafe_allow_html=True)
                        st.write(f"  Confidence: {info['stenosis_confidence']:.2f}")
                        if info['overlapping_segments_names']:
                            segments_str = ', '.join(info['overlapping_segments_names'])
                            st.write(f"  Segment(s): {segments_str}")
                        else:
                            st.write("  No overlapping segments detected.")

                    if not stenosis_info_list:
                        st.warning("No selected stenosis detected to display.")

                else:
                    st.warning("No masks detected in this image.")
            else:
                st.warning("No results detected in the image.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

    # Display the original image in the second column
    with col_original:
        st.subheader("Original Image")
        st.image(image_to_process, use_column_width=True, clamp=True)
