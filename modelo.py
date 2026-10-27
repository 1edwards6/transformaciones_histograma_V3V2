"""Modelo V3 global y V3 local con cuadrícula 2x2, basado en el cuaderno de Colab."""
import math
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator



def encontrar_intervalo_minimo(hist, porcentaje):
    """Intervalo más corto que cubre el porcentaje indicado.

    Desempates: menor ancho, después mayor cantidad de píxeles y luego menor inicio.
    Devuelve r_min, r_max, píxeles dentro, total y píxeles requeridos.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 frecuencias.")
    if np.any(h < 0):
        raise ValueError("Las frecuencias no pueden ser negativas.")
    if not (0 < float(porcentaje) <= 100):
        raise ValueError("El porcentaje debe ser mayor que 0 y como máximo 100.")
    total = int(h.sum())
    if total <= 0:
        raise ValueError("El histograma está vacío.")

    requeridos = int(math.ceil(total * float(porcentaje) / 100.0))
    prefijos = np.concatenate(([0], np.cumsum(h, dtype=np.int64)))
    candidatos = []
    for inicio in range(256):
        objetivo_prefijo = int(prefijos[inicio]) + requeridos
        if objetivo_prefijo > total:
            continue
        fin_exclusivo = int(np.searchsorted(prefijos, objetivo_prefijo, side="left"))
        final = fin_exclusivo - 1
        if final < inicio or final > 255:
            continue
        cantidad = int(prefijos[final + 1] - prefijos[inicio])
        ancho = final - inicio + 1
        candidatos.append((ancho, -cantidad, inicio, final, cantidad))
    if not candidatos:
        raise RuntimeError("No se encontró un intervalo válido.")
    ancho, cantidad_negativa, r_min, r_max, dentro = min(candidatos)
    return int(r_min), int(r_max), int(dentro), total, int(requeridos)


def aproximar_histograma(histograma_rango, metodo="PCHIP", num_puntos=250, num_armonicos=12):
    """Aproxima el histograma seleccionado siguiendo las opciones de la V3."""
    h = np.asarray(histograma_rango, dtype=float).ravel()
    n = len(h)
    coordenadas = np.arange(n, dtype=float)
    metodo = metodo.upper()
    if n == 1:
        return h.copy()

    if metodo in ("PCHIP", "LINEAL"):
        puntos_efectivos = min(max(2, int(num_puntos)), n)
        puntos_x = np.unique(np.rint(np.linspace(0, n - 1, puntos_efectivos)).astype(int))
        puntos_y = h[puntos_x]
        if metodo == "PCHIP":
            interpolante = PchipInterpolator(puntos_x, puntos_y, extrapolate=False)
            aproximado = np.asarray(interpolante(coordenadas), dtype=float)
        else:
            aproximado = np.interp(coordenadas, puntos_x, puntos_y)
    elif metodo == "FOURIER":
        if not 0 <= int(num_armonicos) <= n // 2:
            # Para regiones de tamaño variable limitamos el número de armónicos efectivo.
            num_armonicos = min(max(0, int(num_armonicos)), n // 2)
        coeficientes = np.fft.rfft(h)
        max_armonicos = len(coeficientes) - 1
        k = min(max(0, int(num_armonicos)), max_armonicos)
        coeficientes_filtrados = np.zeros_like(coeficientes)
        coeficientes_filtrados[:k + 1] = coeficientes[:k + 1]
        aproximado = np.fft.irfft(coeficientes_filtrados, n=n)
    else:
        raise ValueError("METODO_APROXIMACION debe ser PCHIP, LINEAL o FOURIER.")
    return np.maximum(np.asarray(aproximado, dtype=float), 0.0)


def frecuencias_realmente_asignadas(frecuencias_objetivo, total_pixeles):
    """Reproduce la regla V3: recorre los niveles de salida y asigna los sobrantes a 255."""
    frecuencias = np.maximum(np.asarray(frecuencias_objetivo, dtype=np.int64).ravel(), 0)
    if frecuencias.size != 256:
        raise ValueError("Se requieren 256 frecuencias objetivo.")
    logradas = np.zeros(256, dtype=np.int64)
    cursor = 0
    total = int(total_pixeles)
    for nivel in range(256):
        disponibles = total - cursor
        if disponibles <= 0:
            break
        cantidad = min(int(frecuencias[nivel]), disponibles)
        if cantidad > 0:
            logradas[nivel] += cantidad
            cursor += cantidad
    if cursor < total:
        logradas[255] += total - cursor
    return logradas


def calcular_objetivo_v3(hist, porcentaje=99.5, metodo="PCHIP", num_puntos=250, num_armonicos=12):
    """Aplica el cálculo de frecuencias objetivo de la V3 a un histograma dado.

    A, B, C y D se calculan desde ese histograma y su número de píxeles.
    Devuelve histograma objetivo, distribución realmente asignada y diagnóstico.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    if h.size != 256 or np.any(h < 0) or h.sum() <= 0:
        raise ValueError("Se requiere un histograma válido de 256 niveles y suma positiva.")
    N = int(h.sum())
    r_min, r_max, pixeles_dentro, total, requeridos = encontrar_intervalo_minimo(h, porcentaje)
    h_rango = h[r_min:r_max + 1].astype(float).copy()
    longitud = len(h_rango)
    coordenadas_locales = np.arange(longitud, dtype=float)
    h_aprox = aproximar_histograma(h_rango, metodo, num_puntos, num_armonicos)

    rd = int(r_max - r_min)
    A = 255.0 / rd if rd > 0 else 1.0
    B = -float(r_min)
    D = 0.0

    niveles_float = np.arange(256, dtype=float)
    coordenada_fuente = (niveles_float - r_min - B) / A
    soporte = (coordenada_fuente >= 0) & (coordenada_fuente <= longitud - 1)
    valores_fuente = np.interp(coordenada_fuente, coordenadas_locales, h_aprox, left=0.0, right=0.0)
    base = np.zeros(256, dtype=float)
    base[soporte] = np.maximum(valores_fuente[soporte], 0.0)
    suma_base = float(base.sum())

    if suma_base <= 0:
        # Salvaguarda si una aproximación patológica produce área nula.
        objetivo = h.copy()
        logradas = frecuencias_realmente_asignadas(objetivo, N)
        diagnostico = {
            "r_min": r_min, "r_max": r_max, "pixeles_dentro": pixeles_dentro,
            "porcentaje_real": 100.0 * pixeles_dentro / N,
            "A": A, "B": B, "C": 1.0, "D": D, "N": N,
            "fallback": "histograma original por área transformada nula",
            "histograma_aproximado": h_aprox,
            "cdf_entrada": np.cumsum(h) / N,
            "cdf_salida": np.cumsum(logradas) / N,
        }
        return objetivo, logradas, diagnostico

    def calcular_frecuencias(c):
        return np.rint(c * base).astype(np.int64)

    def suma_frecuencias(c):
        return int(calcular_frecuencias(c).sum())

    C_area_total = N / suma_base
    inferior = 0.0
    superior = max(C_area_total, np.finfo(float).tiny)
    while suma_frecuencias(superior) < N:
        superior *= 2.0
    for _ in range(200):
        medio = inferior + (superior - inferior) / 2.0
        if medio == inferior or medio == superior:
            break
        if suma_frecuencias(medio) >= N:
            superior = medio
        else:
            inferior = medio
    C = superior
    for _ in range(1000):
        anterior = np.nextafter(C, -np.inf)
        if anterior <= 0 or suma_frecuencias(anterior) < N:
            break
        C = anterior

    funcion_transformada = np.maximum(C * base + D, 0.0)
    objetivo = np.maximum(np.rint(funcion_transformada).astype(np.int64), 0)
    logradas = frecuencias_realmente_asignadas(objetivo, N)

    diagnostico = {
        "r_min": r_min, "r_max": r_max, "pixeles_dentro": pixeles_dentro,
        "porcentaje_real": 100.0 * pixeles_dentro / N,
        "A": A, "B": B, "C": C, "D": D, "N": N,
        "C_area_total": C_area_total, "suma_base": suma_base,
        "suma_objetivo": int(objetivo.sum()),
        "suma_asignada": int(logradas.sum()), "fallback": "",
        "histograma_aproximado": h_aprox,
        "cdf_entrada": np.cumsum(h) / N,
        "cdf_salida": np.cumsum(logradas) / N,
    }
    return objetivo, logradas, diagnostico


def reasignar_pixeles_por_frecuencia(imagen_gris, frecuencias_objetivo):
    """Reasignación ordenada exacta usada por la V3 global."""
    original = np.asarray(imagen_gris, dtype=np.uint8)
    plano = original.ravel()
    total = plano.size
    logradas = frecuencias_realmente_asignadas(frecuencias_objetivo, total)
    orden = np.argsort(plano, kind="stable")
    salida = np.empty_like(plano)
    cursor = 0
    for nivel, cantidad in enumerate(logradas):
        cantidad = int(cantidad)
        if cantidad:
            salida[orden[cursor:cursor + cantidad]] = nivel
            cursor += cantidad
    return salida.reshape(original.shape), np.bincount(salida, minlength=256)

def crear_mapa_intensidad(hist_entrada, hist_salida_logrado, proteger_constante=True):
    """Crea T(r): una sola intensidad de salida por cada intensidad de entrada.

    Regla de desambiguación: entre todos los niveles de salida a los que la V3
    asignaría píxeles de una misma intensidad de entrada, se elige la mediana inferior:
    índice (n - 1) // 2. Para intensidades de entrada no observadas se interpola T.
    """
    h_in = np.asarray(hist_entrada, dtype=np.int64).ravel()
    h_out = np.asarray(hist_salida_logrado, dtype=np.int64).ravel()
    if h_in.size != 256 or h_out.size != 256:
        raise ValueError("Ambos histogramas deben tener 256 niveles.")
    N = int(h_in.sum())
    if N <= 0 or int(h_out.sum()) != N:
        raise ValueError("La suma de los histogramas debe ser el mismo total positivo.")

    niveles_entrada_presentes = np.flatnonzero(h_in > 0)
    if proteger_constante and len(niveles_entrada_presentes) == 1:
        # No hay variación local que realzar; conservar la región constante.
        return np.arange(256, dtype=float), {
            "niveles_entrada_presentes": niveles_entrada_presentes,
            "salidas_posibles_por_intensidad": {int(niveles_entrada_presentes[0]): [int(niveles_entrada_presentes[0])]},
            "representantes": {int(niveles_entrada_presentes[0]): int(niveles_entrada_presentes[0])},
            "region_constante_protegida": True,
            "cantidad_ambiguas": 0,
        }

    # Rango acumulado [inicio, fin) que ocupa cada nivel de salida en la lista ordenada.
    intervalos_salida = []
    cursor = 0
    for nivel_salida, cantidad in enumerate(h_out):
        cantidad = int(cantidad)
        if cantidad > 0:
            intervalos_salida.append((cursor, cursor + cantidad, int(nivel_salida)))
            cursor += cantidad
    if cursor != N:
        raise RuntimeError("Los intervalos de salida no cubren todos los píxeles.")

    cdf_entrada_entera = np.cumsum(h_in, dtype=np.int64)
    salidas_posibles = {}
    representantes = {}
    inicio_entrada = 0
    for nivel_entrada in niveles_entrada_presentes:
        fin_entrada = int(cdf_entrada_entera[nivel_entrada])
        posibles = [
            nivel_salida
            for inicio_salida, fin_salida, nivel_salida in intervalos_salida
            if max(inicio_entrada, inicio_salida) < min(fin_entrada, fin_salida)
        ]
        if not posibles:
            raise RuntimeError(f"No se hallaron salidas para la intensidad {nivel_entrada}.")
        # Los niveles son distintos y ya vienen ordenados de menor a mayor.
        indice_mediana_inferior = (len(posibles) - 1) // 2
        representante = int(posibles[indice_mediana_inferior])
        salidas_posibles[int(nivel_entrada)] = [int(v) for v in posibles]
        representantes[int(nivel_entrada)] = representante
        inicio_entrada = fin_entrada

    x = np.asarray(list(representantes.keys()), dtype=float)
    y = np.asarray(list(representantes.values()), dtype=float)
    mapa = np.interp(np.arange(256, dtype=float), x, y, left=y[0], right=y[-1])
    ambiguas = sum(len(v) > 1 for v in salidas_posibles.values())
    return mapa, {
        "niveles_entrada_presentes": niveles_entrada_presentes,
        "salidas_posibles_por_intensidad": salidas_posibles,
        "representantes": representantes,
        "region_constante_protegida": False,
        "cantidad_ambiguas": int(ambiguas),
    }


def conservar_color(imagen_rgb_original: np.ndarray, imagen_intensidad: np.ndarray) -> np.ndarray:
    """Sustituye la luminancia/intensidad transformada y conserva el color original.

    Replica el procedimiento del cuaderno: convierte RGB a CIELAB, reemplaza L,
    y convierte de nuevo a RGB. En imágenes monocromáticas devuelve escala de grises.
    """
    intensidad = np.asarray(imagen_intensidad, dtype=np.uint8)
    color = np.asarray(imagen_rgb_original, dtype=np.uint8)
    es_color = np.any(np.max(color, axis=2) != np.min(color, axis=2))
    if not es_color:
        return intensidad
    imagen_lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)
    imagen_lab[:, :, 0] = intensidad
    return cv2.cvtColor(imagen_lab, cv2.COLOR_LAB2RGB)


def procesar_imagen_rgb(
    imagen_rgb_original: np.ndarray,
    porcentaje_conservado: float = 92.5,
    metodo_aproximacion: str = "PCHIP",
    num_puntos: int = 150,
    num_armonicos: int = 12,
    clahe_limite: float = 2.0,
    clahe_cuadricula: Tuple[int, int] = (8, 8),
) -> Dict[str, np.ndarray]:
    """Calcula los cuatro métodos y devuelve sus resultados listos para visualizar/descargar.

    La cuadrícula de V3 local está fijada en 2x2, como en la versión actual del cuaderno.
    Los resultados en imágenes RGB conservan la crominancia de la entrada, igual que la
    función conservar_color utilizada para las visualizaciones del cuaderno.
    """
    color_original = np.asarray(imagen_rgb_original, dtype=np.uint8)
    if color_original.ndim != 3 or color_original.shape[2] != 3:
        raise ValueError("La imagen de entrada debe estar en formato RGB.")
    imagen = np.asarray(Image.fromarray(color_original).convert("L"), dtype=np.uint8)
    filas_grid = 2
    columnas_grid = 2

    # V3 GLOBAL: misma lógica de objetivo y reasignación aplicada a toda la imagen.
    histograma_global = np.bincount(imagen.ravel(), minlength=256).astype(np.int64)
    objetivo_global, logrado_global, diagnostico_global = calcular_objetivo_v3(
        histograma_global,
        porcentaje_conservado,
        metodo_aproximacion,
        num_puntos,
        num_armonicos,
    )
    imagen_v3_global, _ = reasignar_pixeles_por_frecuencia(imagen, objetivo_global)

    alto, ancho = imagen.shape
    if filas_grid > alto or columnas_grid > ancho:
        raise ValueError("La imagen debe tener al menos 2 píxeles de alto y de ancho para usar la cuadrícula 2×2.")

    # Regiones equilibradas, igual que en el cuaderno.
    filas = np.array_split(np.arange(alto), filas_grid)
    columnas = np.array_split(np.arange(ancho), columnas_grid)
    centros_y = np.array([grupo.mean() for grupo in filas], dtype=float)
    centros_x = np.array([grupo.mean() for grupo in columnas], dtype=float)
    mapas_locales = np.empty((filas_grid, columnas_grid, 256), dtype=float)
    frecuencias_objetivo_locales = np.zeros((filas_grid, columnas_grid, 256), dtype=np.int64)
    frecuencias_logradas_locales = np.zeros_like(frecuencias_objetivo_locales)
    cdfs_entrada_locales = np.zeros((filas_grid, columnas_grid, 256), dtype=float)
    cdfs_salida_locales = np.zeros_like(cdfs_entrada_locales)

    for i, ys in enumerate(filas):
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        for j, xs in enumerate(columnas):
            x0, x1 = int(xs[0]), int(xs[-1]) + 1
            region = imagen[y0:y1, x0:x1]
            h = np.bincount(region.ravel(), minlength=256).astype(np.int64)
            objetivo, logrado, diag = calcular_objetivo_v3(
                h, porcentaje_conservado, metodo_aproximacion, num_puntos, num_armonicos
            )
            mapa, _detalle_mapa = crear_mapa_intensidad(
                h, logrado, proteger_constante=True
            )
            mapas_locales[i, j] = mapa
            frecuencias_objetivo_locales[i, j] = objetivo
            frecuencias_logradas_locales[i, j] = logrado
            cdfs_entrada_locales[i, j] = np.cumsum(h) / h.sum()
            cdfs_salida_locales[i, j] = np.cumsum(logrado) / logrado.sum()

    # Interpolación espacial bilineal de los mapas locales.
    def indices_y_pesos_interpolacion(coordenadas, centros):
        coordenadas = np.asarray(coordenadas, dtype=float)
        centros = np.asarray(centros, dtype=float)
        if len(centros) == 1:
            ceros = np.zeros(coordenadas.shape, dtype=int)
            return ceros, ceros, np.zeros(coordenadas.shape, dtype=float)
        idx_alto = np.searchsorted(centros, coordenadas, side="right")
        idx_alto = np.clip(idx_alto, 0, len(centros) - 1)
        idx_bajo = np.maximum(idx_alto - 1, 0)
        denom = centros[idx_alto] - centros[idx_bajo]
        peso_alto = np.zeros(coordenadas.shape, dtype=float)
        validos = denom > 0
        peso_alto[validos] = (coordenadas[validos] - centros[idx_bajo[validos]]) / denom[validos]
        peso_alto = np.clip(peso_alto, 0.0, 1.0)
        peso_alto[coordenadas <= centros[0]] = 0.0
        peso_alto[coordenadas >= centros[-1]] = 1.0
        idx_bajo[coordenadas <= centros[0]] = 0
        idx_alto[coordenadas <= centros[0]] = 0
        idx_bajo[coordenadas >= centros[-1]] = len(centros) - 1
        idx_alto[coordenadas >= centros[-1]] = len(centros) - 1
        peso_alto[coordenadas >= centros[-1]] = 0.0
        return idx_bajo, idx_alto, peso_alto

    idx_y0, idx_y1, peso_y = indices_y_pesos_interpolacion(np.arange(alto), centros_y)
    idx_x0, idx_x1, peso_x = indices_y_pesos_interpolacion(np.arange(ancho), centros_x)
    imagen_v3_local = np.empty_like(imagen)
    # Procesamos por bandas para no crear cuatro matrices float del tamaño de toda la imagen.
    # La fórmula es la misma que en el cuaderno, pero con menor pico de memoria.
    tamano_banda = 128
    for inicio_y in range(0, alto, tamano_banda):
        fin_y = min(inicio_y + tamano_banda, alto)
        ys_banda = np.arange(inicio_y, fin_y)
        intensidades = imagen[inicio_y:fin_y].astype(np.int64)
        T00 = mapas_locales[idx_y0[ys_banda, None], idx_x0[None, :], intensidades]
        T01 = mapas_locales[idx_y0[ys_banda, None], idx_x1[None, :], intensidades]
        T10 = mapas_locales[idx_y1[ys_banda, None], idx_x0[None, :], intensidades]
        T11 = mapas_locales[idx_y1[ys_banda, None], idx_x1[None, :], intensidades]
        wy = peso_y[ys_banda, None]
        wx = peso_x[None, :]
        transformacion = (
            (1 - wy) * (1 - wx) * T00
            + (1 - wy) * wx * T01
            + wy * (1 - wx) * T10
            + wy * wx * T11
        )
        imagen_v3_local[inicio_y:fin_y] = np.clip(np.rint(transformacion), 0, 255).astype(np.uint8)

    imagen_he = cv2.equalizeHist(imagen)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_limite), tileGridSize=tuple(clahe_cuadricula))
    imagen_clahe = clahe.apply(imagen)

    # Devolvemos las variantes coloreadas de acuerdo con la presentación del cuaderno.
    return {
        "V3 Global": conservar_color(color_original, imagen_v3_global),
        "V3 Local (2×2)": conservar_color(color_original, imagen_v3_local),
        "CLAHE": conservar_color(color_original, imagen_clahe),
        "HE": conservar_color(color_original, imagen_he),
    }
