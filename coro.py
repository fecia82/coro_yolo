import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import itertools

# Configuración de la página
st.set_page_config(page_title="Detección Automática de Estenosis Coronaria (YOLO)", layout="wide")

# Indicador de opciones en la barra lateral
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Las opciones están disponibles en la barra lateral.</p>",
    unsafe_allow_html=True
)

# Cargar los modelos YOLO una vez al inicio
@st.cache_resource
def load_models():
    model_stenosis_path = "model.pt"  # Modelo de estenosis
    model_segments_path = "model_s.pt"  # Modelo de segmentos
    if not os.path.exists(model_stenosis_path):
        st.error(f"El modelo de estenosis no se encontró en la ruta especificada: {model_stenosis_path}")
        st.stop()
    if not os.path.exists(model_segments_path):
        st.error(f"El modelo de segmentos no se encontró en la ruta especificada: {model_segments_path}")
        st.stop()
    model_stenosis = YOLO(model_stenosis_path)
    model_segments = YOLO(model_segments_path)
    return model_stenosis, model_segments

model_stenosis, model_segments = load_models()

# Barra lateral para controles
st.sidebar.header("Configuración")

# Slider para umbral de confianza
threshold = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.3, 0.05)

# Slider para transparencia de la máscara de estenosis
transparency_stenosis = st.sidebar.slider("Transparencia de la máscara de estenosis", 0.0, 1.0, 0.4, 0.05)

# Slider para transparencia de los contornos de segmentos
transparency_segments = st.sidebar.slider("Transparencia de los contornos de segmentos", 0.0, 1.0, 0.3, 0.05)

# Función para cargar imágenes de prueba
def load_test_images():
    test_images = {}
    test_image_names = ["1.png", "2.png", "3.png"]  # Asegúrate de que sean PNG
    for img_name in test_image_names:
        if os.path.exists(img_name):
            image = Image.open(img_name).convert("RGB")
            test_images[img_name] = image
        else:
            st.warning(f"No se encontró la imagen de prueba: {img_name}")
    return test_images

# Callback para seleccionar una imagen de prueba
def select_test_image(img_name):
    st.session_state['selected_test_image'] = img_name

# Inicializar el estado de la imagen seleccionada
if 'selected_test_image' not in st.session_state:
    st.session_state['selected_test_image'] = None

# Sección principal: Subir una imagen o seleccionar una imagen de prueba

# Opción para subir una imagen
uploaded_file = st.file_uploader("Sube tu propia imagen", type=["png", "jpg", "jpeg"])

# Variable para almacenar la imagen a procesar
image_to_process = None

if uploaded_file is not None:
    # Leer la imagen subida
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_to_process = image
elif st.session_state['selected_test_image'] is not None:
    # Cargar la imagen de prueba seleccionada
    selected_image = Image.open(st.session_state['selected_test_image']).convert("RGB")
    image_np = np.array(selected_image)
    image_to_process = selected_image

# Mostrar imágenes de prueba solo si no se ha subido o seleccionado una imagen
if uploaded_file is None and st.session_state['selected_test_image'] is None:
    test_images = load_test_images()
    if test_images:
        st.subheader("O elige una imagen de prueba:")
        cols = st.columns(3)
        for idx, (img_name, img) in enumerate(test_images.items()):
            with cols[idx]:
                # Crear una copia de la imagen para la miniatura
                thumbnail = img.copy()
                thumbnail.thumbnail((int(img.width * 0.3), int(img.height * 0.3)))  # 30% del tamaño original
                # Mostrar la imagen como miniatura
                st.image(thumbnail, use_column_width=False, clamp=True)
                # Hacer la imagen clicable usando un botón debajo
                # Asignar una clave única a cada botón para evitar duplicaciones
                st.button(f"Seleccionar {img_name}", key=f"select_test_{img_name}", on_click=select_test_image, args=(img_name,))
else:
    # Si se ha subido o seleccionado una imagen, ocultar las imágenes de prueba
    pass

# Definición de los nombres de segmentos
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

# Procesar la imagen si está disponible
if image_to_process is not None:
    try:
        # Crear dos columnas para mostrar imágenes lado a lado
        # Primera columna: Imagen con Máscaras
        # Segunda columna: Imagen Original
        col_masks, col_original = st.columns(2)

        with col_masks:
            st.subheader("Máscaras de Estenosis y Segmentos")
            # Realizar inferencia con ambos modelos
            with st.spinner('Procesando...'):
                results_stenosis = model_stenosis.predict(source=image_np, conf=threshold, save=False, task='segment')
                results_segments = model_segments.predict(source=image_np, conf=threshold, save=False, task='segment')

            # Procesar resultados
            if results_stenosis and results_segments:
                result_stenosis = results_stenosis[0]  # Asumiendo que solo se procesa una imagen
                result_segments = results_segments[0]

                if (result_stenosis.masks is not None and len(result_stenosis.masks.data) > 0 and
                    result_segments.masks is not None and len(result_segments.masks.data) > 0):

                    # Obtener máscaras y clases
                    masks_stenosis = result_stenosis.masks.data.cpu().numpy()  # (num_masks, height, width)
                    confidences_stenosis = result_stenosis.boxes.conf.cpu().numpy()  # Confianzas
                    classes_stenosis = result_stenosis.boxes.cls.cpu().numpy().astype(int)
                    names_stenosis = model_stenosis.names  # Nombres de clases

                    masks_segments = result_segments.masks.data.cpu().numpy()
                    confidences_segments = result_segments.boxes.conf.cpu().numpy()
                    classes_segments = result_segments.boxes.cls.cpu().numpy().astype(int)
                    names_segments = model_segments.names

                    # Asignar colores únicos a cada máscara de estenosis
                    color_palette_stenosis = [
                        (255, 0, 0),      # Rojo
                        (255, 165, 0),    # Naranja
                        (255, 255, 0),    # Amarillo
                        (255, 69, 0),     # Rojo anaranjado
                        (255, 215, 0),    # Oro
                        (255, 99, 71),    # Tomate
                        (255, 140, 0),    # Dark orange
                        (255, 127, 80),    # Coral
                        (255, 105, 180),  # Hot pink
                        (255, 20, 147),    # Deep pink
                        (255, 0, 255),    # Magenta
                        (238, 130, 238),  # Violet
                    ]
                    color_cycle_stenosis = itertools.cycle(color_palette_stenosis)

                    mask_colors_stenosis = {}
                    for idx_mask in range(len(masks_stenosis)):
                        mask_colors_stenosis[idx_mask] = next(color_cycle_stenosis)

                    # Asignar colores únicos a cada máscara de segmentos
                    color_palette_segments = [
                        (0, 0, 255),      # Azul
                        (0, 255, 0),      # Verde
                        (0, 255, 255),    # Cian
                        (138, 43, 226),   # Blue violet
                        (75, 0, 130),     # Indigo
                        (0, 128, 128),    # Teal
                        (34, 139, 34),    # Forest green
                        (255, 182, 193),  # Light pink
                        (173, 216, 230),  # Light blue
                        (0, 100, 0),      # Dark green
                        (128, 0, 128),    # Purple
                        (255, 0, 0),      # Red (puede superponerse, ajustar si es necesario)
                    ]
                    color_cycle_segments = itertools.cycle(color_palette_segments)

                    mask_colors_segments = {}
                    for idx_mask in range(len(masks_segments)):
                        mask_colors_segments[idx_mask] = next(color_cycle_segments)

                    # Crear una copia de la imagen original para superponer las máscaras
                    img_with_masks = image_np.copy()

                    # Lista para almacenar información de máscaras de estenosis
                    stenosis_info_list = []

                    # Preprocesar todas las estenosis para obtener la información de segmentación
                    stenosis_overlaps = []

                    for idx_stenosis, (cls, conf) in enumerate(zip(classes_stenosis, confidences_stenosis)):
                        mask_stenosis = masks_stenosis[idx_stenosis]
                        mask_stenosis_resized = cv2.resize(mask_stenosis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # Determinar segmentos superpuestos
                        overlapping_segments = set()
                        for idx_segment, mask_segment in enumerate(masks_segments):
                            mask_segment_resized = cv2.resize(mask_segment, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                            # Calcular superposición
                            intersection = np.logical_and(mask_stenosis_resized > 0.5, mask_segment_resized > 0.5)
                            if np.any(intersection):
                                class_segment = classes_segments[idx_segment]
                                segment_number = names_segments[class_segment]
                                segment_name = segment_number_to_name.get(segment_number, 'Desconocido')
                                overlapping_segments.add(segment_name)

                        stenosis_overlaps.append(overlapping_segments)

                    # Barra lateral para seleccionar máscaras de estenosis con indicación de color
                    st.sidebar.subheader("Estenosis Localizadas")

                    for idx_stenosis, (cls, conf) in enumerate(zip(classes_stenosis, confidences_stenosis)):
                        color_rgb = mask_colors_stenosis[idx_stenosis]
                        color_hex = '#%02x%02x%02x' % color_rgb
                        # Crear un pequeño cuadro de color usando markdown
                        color_box = f'<div style="width: 15px; height: 15px; background-color: {color_hex}; display: inline-block; margin-right: 5px;"></div>'
                        overlapping_segments = stenosis_overlaps[idx_stenosis]
                        # Formatear los segmentos superpuestos
                        if overlapping_segments:
                            segments_str = ', '.join(overlapping_segments)
                        else:
                            segments_str = 'N/A'
                        # Formatear la etiqueta
                        label = f"{segments_str} (Confianza: {conf:.2f})"
                        # Crear una columna para el color y otra para el checkbox
                        col_color, col_checkbox = st.sidebar.columns([1, 10])
                        with col_color:
                            st.markdown(color_box, unsafe_allow_html=True)
                        with col_checkbox:
                            # Usar una clave única para cada checkbox
                            mask_selection = st.checkbox(label, value=True, key=f"mask_checkbox_{idx_stenosis}")

                        if mask_selection:
                            # Procesar la máscara de estenosis
                            mask_stenosis = masks_stenosis[idx_stenosis]
                            mask_stenosis_resized = cv2.resize(mask_stenosis, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                            # Superponer la máscara de estenosis en la imagen
                            color_mask = np.zeros_like(image_np)
                            color_mask[mask_stenosis_resized > 0.5] = color_rgb
                            img_with_masks = cv2.addWeighted(img_with_masks, 1.0, color_mask, transparency_stenosis, 0)

                            # Almacenar la información
                            # Asumiendo que cada estenosis corresponde a al menos un segmento
                            stenosis_class = list(overlapping_segments)[0] if overlapping_segments else 'N/A'
                            stenosis_info = {
                                'stenosis_index': idx_stenosis,
                                'stenosis_class': stenosis_class,
                                'stenosis_confidence': conf,
                                'overlapping_segments_names': overlapping_segments,
                                'color_rgb': color_rgb,
                            }
                            stenosis_info_list.append(stenosis_info)

                    # Barra lateral para seleccionar máscaras de segmentos con indicación de color
                    st.sidebar.subheader("Segmentos Identificados")

                    for idx_segment, (cls_seg, conf_seg) in enumerate(zip(classes_segments, confidences_segments)):
                        color_rgb_seg = mask_colors_segments[idx_segment]
                        color_hex_seg = '#%02x%02x%02x' % color_rgb_seg
                        # Crear un pequeño cuadro de contorno usando CSS
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
                        segment_description = segment_number_to_name.get(segment_number, 'Desconocido')
                        label_seg = f"◻️ {segment_description} (Confianza: {conf_seg:.2f})"
                        # Crear una columna para el contorno y otra para el checkbox
                        col_color_seg, col_checkbox_seg = st.sidebar.columns([1, 10])
                        with col_color_seg:
                            st.markdown(color_box_seg, unsafe_allow_html=True)
                        with col_checkbox_seg:
                            # Usar una clave única para cada checkbox y deseleccionarlas por defecto
                            mask_selection_seg = st.checkbox(label_seg, value=False, key=f"mask_checkbox_seg_{idx_segment}")

                        if mask_selection_seg:
                            # Procesar la máscara de segmento
                            mask_segment = masks_segments[idx_segment]
                            mask_segment_resized = cv2.resize(mask_segment, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                            # Encontrar contornos de la máscara
                            mask_uint8 = (mask_segment_resized > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Dibujar contornos más finos en la imagen
                            cv2.drawContours(img_with_masks, contours, -1, color_rgb_seg, 1)  # Grosor = 1

                    # Mostrar la imagen con máscaras
                    st.image(img_with_masks, use_column_width=True, clamp=True)

                    # Mostrar la información de estenosis y segmentos
                    st.subheader("Resultados de Estenosis y Segmentos")
                    for info in stenosis_info_list:
                        color_rgb = info['color_rgb']
                        color_hex = '#%02x%02x%02x' % color_rgb
                        color_box = f'<div style="width: 15px; height: 15px; background-color: {color_hex}; display: inline-block; margin-right: 5px;"></div>'
                        st.markdown(f"{color_box} **{info['stenosis_class']}**", unsafe_allow_html=True)
                        st.write(f"  Confianza: {info['stenosis_confidence']:.2f}")
                        if info['overlapping_segments_names']:
                            segments_str = ', '.join(info['overlapping_segments_names'])
                            st.write(f"  Segmento(s): {segments_str}")
                        else:
                            st.write("  No se detectaron segmentos superpuestos.")

                    if not stenosis_info_list:
                        st.warning("No se detectaron estenosis seleccionadas para mostrar.")

                else:
                    st.warning("No se detectaron máscaras en esta imagen.")
            else:
                st.warning("No se detectaron resultados en la imagen.")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")

    # Mostrar la imagen original en la segunda columna
    with col_original:
        st.subheader("Imagen Original")
        st.image(image_to_process, use_column_width=True, clamp=True)
