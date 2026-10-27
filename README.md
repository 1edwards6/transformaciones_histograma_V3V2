# Transformación de histogramas: V3 global y V3 local (2×2)

Aplicación web en Streamlit basada en el cuaderno de Colab adjunto. Permite cargar una imagen, comparar cuatro resultados y descargar **solo el método elegido**.

## Métodos disponibles

- **V3 Global:** aplica el modelo V3 al histograma de toda la imagen.
- **V3 Local (2×2):** calcula las transformaciones por región en una cuadrícula fija de cuatro regiones y las combina mediante interpolación bilineal.
- **HE:** ecualización global clásica del histograma.
- **CLAHE:** ecualización adaptativa limitada por contraste.

En imágenes a color, la app sigue la presentación visual del cuaderno: calcula las transformaciones sobre la intensidad en escala de grises y vuelve a componer el resultado en CIELAB, reemplazando el canal L para conservar la crominancia original.

## Ejecutar localmente

Se recomienda Python 3.10 o posterior.

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instala dependencias y ejecuta la aplicación:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador. Carga una imagen y espera a que se calculen los cuatro métodos. Debajo de la comparación, usa la lista **¿Qué imagen deseas descargar?** para escoger `V3 Global`, `V3 Local (2×2)`, `CLAHE` o `HE`; el botón descarga únicamente esa imagen en PNG.

## Publicar en GitHub y desplegar en Streamlit Community Cloud

1. Descomprime este proyecto y sube los archivos a un repositorio de GitHub.
2. En Streamlit Community Cloud, crea una app nueva y selecciona el repositorio.
3. Indica `app.py` como archivo principal.
4. La plataforma instalará las dependencias de `requirements.txt`.

## Parámetros predeterminados

- Porcentaje de intervalo: `92.5 %`
- Aproximación: `PCHIP`
- Puntos de aproximación: `150`
- Armónicos de Fourier: `12`
- Cuadrícula de V3 local: `2 × 2` (fija)
- CLAHE: límite `2.0`, cuadrícula interna `(8, 8)`

Los parámetros disponibles se pueden cambiar en la barra lateral. La cuadrícula V3 local permanece en 2×2, tal como en el cuaderno de referencia.

## Cuaderno de referencia

La carpeta `notebooks/` incluye una copia del cuaderno de Colab recibido para conservar el procedimiento de referencia y facilitar la trazabilidad de la implementación.
