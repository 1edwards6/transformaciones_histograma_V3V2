from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from modelo import procesar_imagen_rgb

st.set_page_config(
    page_title="Transformación de histogramas | V3 local",
    page_icon="🖼️",
    layout="wide",
)

st.title("Transformación de histogramas: V3 global y V3 local")
st.write(
    "Compara V3 global, V3 local con cuadrícula 2×2, ecualización global (HE) "
    "y CLAHE. Escoge un solo resultado en la lista desplegable para previsualizarlo "
    "y descargarlo individualmente."
)

with st.sidebar:
    st.header("Parámetros del modelo")
    porcentaje = st.slider(
        "Porcentaje de píxeles conservado en el intervalo",
        min_value=70.0,
        max_value=100.0,
        value=92.5,
        step=0.5,
        help="Valor predeterminado del cuaderno actual: 92,5 %. Se aplica a V3 global y a cada región de V3 local.",
    )
    metodo = st.selectbox(
        "Método de aproximación del histograma",
        options=["PCHIP", "LINEAL", "FOURIER"],
        index=0,
    )
    if metodo in ("PCHIP", "LINEAL"):
        num_puntos = st.slider(
            "Puntos de aproximación",
            min_value=2,
            max_value=256,
            value=150,
            step=1,
            help="Parámetro usado por PCHIP o LINEAL.",
        )
    else:
        num_puntos = 150
    if metodo == "FOURIER":
        num_armonicos = st.slider(
            "Número de armónicos",
            min_value=0,
            max_value=128,
            value=12,
            step=1,
        )
    else:
        num_armonicos = 12
    clahe_limite = st.number_input(
        "Límite de CLAHE",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    st.caption("V3 local utiliza una cuadrícula fija de 2 × 2, tal como en el cuaderno actual.")

archivo = st.file_uploader(
    "Carga una imagen para analizar",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    help="La imagen original no se modifica. Los resultados se generan al procesarla.",
)

if archivo is None:
    st.info("Carga una imagen para ver la comparación y habilitar la descarga de resultados.")
    st.stop()

try:
    imagen_pil = ImageOps.exif_transpose(Image.open(BytesIO(archivo.getvalue()))).convert("RGB")
    imagen_rgb = np.asarray(imagen_pil, dtype=np.uint8)
except (UnidentifiedImageError, OSError, ValueError):
    st.error("No se pudo leer el archivo como imagen. Prueba con PNG, JPG, TIFF, BMP o WEBP.")
    st.stop()

if min(imagen_rgb.shape[:2]) < 2:
    st.error("La imagen debe tener al menos 2 píxeles de alto y de ancho para aplicar la cuadrícula 2×2.")
    st.stop()

@st.cache_data(show_spinner=False, max_entries=4)
def procesar_en_cache(
    contenido_imagen: bytes,
    porcentaje_conservado: float,
    metodo_aproximacion: str,
    puntos: int,
    armonicos: int,
    limite_clahe: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with ImageOps.exif_transpose(Image.open(BytesIO(contenido_imagen))) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    resultados = procesar_imagen_rgb(
        rgb,
        porcentaje_conservado=porcentaje_conservado,
        metodo_aproximacion=metodo_aproximacion,
        num_puntos=puntos,
        num_armonicos=armonicos,
        clahe_limite=limite_clahe,
        clahe_cuadricula=(8, 8),
    )
    return rgb, resultados

try:
    with st.spinner("Calculando V3 global, V3 local, HE y CLAHE…"):
        original_rgb, resultados = procesar_en_cache(
            archivo.getvalue(),
            float(porcentaje),
            metodo,
            int(num_puntos),
            int(num_armonicos),
            float(clahe_limite),
        )
except Exception as exc:
    st.error(f"Ocurrió un problema al procesar la imagen: {exc}")
    st.stop()

st.subheader("Comparación de resultados")
columnas = st.columns(5)
with columnas[0]:
    st.image(original_rgb, caption="Original", use_container_width=True)
for columna, nombre in zip(columnas[1:], ["V3 Global", "V3 Local (2×2)", "HE", "CLAHE"]):
    with columna:
        st.image(resultados[nombre], caption=nombre, use_container_width=True)

st.divider()
st.subheader("Descargar un resultado")
metodo_descarga = st.selectbox(
    "¿Qué imagen deseas descargar?",
    options=["V3 Global", "V3 Local (2×2)", "CLAHE", "HE"],
    index=1,
    key="metodo_descarga",
)
imagen_seleccionada = resultados[metodo_descarga]
st.image(imagen_seleccionada, caption=f"Vista previa: {metodo_descarga}", use_container_width=True)

buffer = BytesIO()
Image.fromarray(np.asarray(imagen_seleccionada, dtype=np.uint8)).save(buffer, format="PNG")
slug = {
    "V3 Global": "v3_global",
    "V3 Local (2×2)": "v3_local_2x2",
    "CLAHE": "clahe",
    "HE": "he",
}[metodo_descarga]

st.download_button(
    label=f"Descargar solo {metodo_descarga} (.png)",
    data=buffer.getvalue(),
    file_name=f"resultado_{slug}.png",
    mime="image/png",
    use_container_width=True,
)
st.caption(
    "La descarga contiene únicamente el método seleccionado. Para comparar otro método, "
    "cámbialo en la lista desplegable y usa el mismo botón."
)
