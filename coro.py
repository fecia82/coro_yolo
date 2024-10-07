import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
import itertools

# Page configuration
st.set_page_config(page_title="Coronary stenosis automatic detection (YOLO)", layout="wide")

# Add an indication that the options are in the sidebar
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Options are available in the sidebar.</p>",
    unsafe_allow_html=True
)

# Load the YOLO model once at the start
@st.cache_resource
def load_model():
    model_path = "model.pt"  # Make sure the model is in the same directory or provide the correct path
    if not os.path.exists(model_path):
        st.error(f"The model was not found at the specified path: {model_path}")
        st.stop()
    return YOLO(model_path)

model = load_model()

# Sidebar for controls
st.sidebar.header("Settings")

# Slider for confidence threshold
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Slider for mask transparency
transparency = st.sidebar.slider("Mask Transparency", 0.0, 1.0, 0.2, 0.05)

# Function to load test images
def load_test_images():
    test_images = {}
    test_image_names = ["1.png", "2.png", "3.png"]  # Ensure they are PNG
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

# Initialize the selected image state
if 'selected_test_image' not in st.session_state:
    st.session_state['selected_test_image'] = None

# Main section: Upload an image or select a test image

# Option to upload an image
uploaded_file = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg"])

# Variable to store the image to be processed
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

# Display test images only if no image has been uploaded or selected
if uploaded_file is None and st.session_state['selected_test_image'] is None:
    test_images = load_test_images()
    if test_images:
        st.subheader("Or choose a test image:")
        cols = st.columns(3)
        for idx, (img_name, img) in enumerate(test_images.items()):
            with cols[idx]:
                # Create a copy of the image for the thumbnail
                thumbnail = img.copy()
                thumbnail.thumbnail((int(img.width * 0.3), int(img.height * 0.3)))  # 30% of the original size
                # Display the image as a thumbnail
                st.image(thumbnail, use_column_width=False, clamp=True)
                # Make the image clickable using a button below
                # Assign a unique key to each button to avoid duplications
                st.button(f"Select {img_name}", key=f"select_test_{img_name}", on_click=select_test_image, args=(img_name,))
else:
    # If an image has been uploaded or selected, hide the test images
    pass

# Process the image if available
if image_to_process is not None:
    try:
        # Create two columns to display images side by side
        # First column: Image with Masks
        # Second column: Original Image
        col_masks, col_original = st.columns(2)

        with col_masks:
            st.subheader("Stenosis masks")
            # Perform inference
            with st.spinner('Processing...'):
                results = model.predict(source=image_np, conf=threshold, save=False)

            # Process results
            if results:
                result = results[0]  # Assuming only one image is processed
                if result.masks is not None and len(result.masks.data) > 0:
                    masks = result.masks.data.cpu().numpy()  # (num_masks, height, width)
                    confidences = result.boxes.conf.cpu().numpy()  # Probabilities
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    names = model.names  # Class names

                    # Assign unique colors to each mask using a predefined palette
                    color_palette = [
                        (255, 0, 0),      # Red
                        (0, 255, 0),      # Green
                        (0, 0, 255),      # Blue
                        (255, 255, 0),    # Yellow
                        (255, 0, 255),    # Magenta
                        (0, 255, 255),    # Cyan
                        (128, 0, 0),      # Maroon
                        (0, 128, 0),      # Dark Green
                        (0, 0, 128),      # Dark Blue
                        (128, 128, 0),    # Olive
                        (128, 0, 128),    # Purple
                        (0, 128, 128),    # Teal
                    ]
                    color_cycle = itertools.cycle(color_palette)

                    # Assign colors to each mask and store them for consistency
                    mask_colors = {}
                    for idx_mask in range(len(masks)):
                        mask_colors[idx_mask] = next(color_cycle)

                    # Create a blank image for the masks
                    mask_image = np.zeros_like(image_np, dtype=np.uint8)

                    # Create a matrix to track which mask is assigned to each pixel
                    mask_indices = np.full((image_np.shape[0], image_np.shape[1]), -1, dtype=int)

                    # List to store mask information
                    mask_info_list = []

                    # Sidebar to select masks with color indication
                    st.sidebar.subheader("Select masks to display")

                    # Use columns to display color and checkbox side by side
                    for idx_mask, (cls, conf) in enumerate(zip(classes, confidences)):
                        color_rgb = mask_colors[idx_mask]
                        color_hex = '#%02x%02x%02x' % color_rgb
                        # Create a small color box using markdown
                        color_box = f'<div style="width: 15px; height: 15px; background-color: {color_hex}; display: inline-block; margin-right: 5px;"></div>'
                        label = f"{names[cls]} - Confidence: {conf:.2f}"
                        # Create one column for the color and another for the checkbox
                        col_color, col_checkbox = st.sidebar.columns([1, 10])
                        with col_color:
                            st.markdown(color_box, unsafe_allow_html=True)
                        with col_checkbox:
                            # Use a unique key for each checkbox
                            mask_selection = st.checkbox(label, value=True, key=f"mask_checkbox_{idx_mask}")

                        # Save the selection state
                        if mask_selection:
                            # Process the mask
                            mask_resized = cv2.resize(masks[idx_mask], (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                            mask_uint8 = (mask_resized * 255).astype(np.uint8)
                            blurred_mask = cv2.GaussianBlur(mask_uint8, (7, 7), 0)
                            mask_smoothed = blurred_mask.astype(np.float32) / 255.0

                            mask_bool = mask_smoothed > 0.3  # Threshold to binarize
                            update_mask = np.logical_and(mask_bool, mask_indices == -1)
                            if not np.any(update_mask):
                                continue

                            mask_indices[update_mask] = idx_mask

                            # Get the color assigned to the mask
                            color_rgb_np = np.array(color_rgb, dtype=np.uint8)  # RGB
                            # Assign colors to the corresponding pixels
                            mask_image[update_mask] = color_rgb_np

                            # Store mask information
                            mask_info = {
                                'mask_index': idx_mask,
                                'class': names[cls],
                                'confidence': conf,
                                'color_rgb': color_rgb,
                            }
                            mask_info_list.append(mask_info)

                    # Overlay the mask image on the original image
                    # Convert mask_image to float32 and scale to [0,1]
                    mask_image_float = mask_image.astype(np.float32) / 255.0
                    # Multiply by transparency
                    mask_image_transparent = mask_image_float * transparency
                    # Convert image_np to float32 and scale to [0,1]
                    image_float = image_np.astype(np.float32) / 255.0
                    # Overlay
                    img_with_masks = image_float + mask_image_transparent
                    # Clip to [0,1] and convert back to uint8
                    img_with_masks = np.clip(img_with_masks, 0, 1) * 255
                    img_with_masks = img_with_masks.astype(np.uint8)

                    # Display the image with masks (RGB)
                    st.image(img_with_masks, use_column_width=True, clamp=True)

                    # Create the legend
                    if mask_info_list:
                        legend_elements = []
                        for mask_info in mask_info_list:
                            color_rgb = mask_info['color_rgb']
                            label = f"{mask_info['class']} - Confidence: {mask_info['confidence']:.2f}"
                            legend_elements.append((color_rgb, label))

                        # Create a legend image
                        legend_fig, legend_ax = plt.subplots(figsize=(6, len(legend_elements)*0.3))
                        legend_ax.axis('off')
                        for idx_leg, (color, label) in enumerate(legend_elements):
                            # Reverse y-position to list top to bottom
                            y_pos = len(legend_elements) - idx_leg -1
                            # Convert color to float for Matplotlib
                            color_normalized = np.array(color) / 255.0
                            legend_ax.add_patch(Rectangle((0, y_pos*0.3), 0.2, 0.2, facecolor=color_normalized))
                            legend_ax.text(0.25, y_pos*0.3 + 0.1, label, verticalalignment='center', fontsize=12)
                        legend_buf = BytesIO()
                        legend_fig.savefig(legend_buf, format="png", bbox_inches='tight')
                        legend_buf.seek(0)
                        st.image(legend_buf, caption="Legend", use_column_width=False)
                        plt.close(legend_fig)
                else:
                    st.warning("No results detected in the image.")
            else:
                st.warning("No masks detected in this image.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

    # Display the original image in the second column
    with col_original:
        st.subheader("Original Image")
        st.image(image_to_process, use_column_width=True, clamp=True)
