"""
limpieza.py — Auditoría y limpieza de los datasets de TechLogistics S.A.S.

Cada dataset tiene su propia función `limpiar_*` que devuelve el DataFrame
limpio junto con un log de decisiones (qué se hizo, a cuántas filas afectó
y por qué). La idea es que ninguna transformación quede sin rastro:
el log alimenta el reporte de limpieza descargable desde la app.

Criterios generales de imputación:
- Mediana para variables numéricas con outliers o distribuciones sesgadas
  (costos, stock, tiempos), porque la media se contamina con los extremos.
- Moda / categoría explícita ("Desconocido") para variables categóricas,
  para no inventar información que no existe.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Carpeta de datos por defecto (relativa a la raíz del proyecto)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EDAD_MAXIMA = 100

EXTREMO_TIEMPO_ENTREGA = 999
EXTREMO_RATING = 99


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_datos(data_dir: Path = DATA_DIR) -> dict:
    """Lee los tres CSV crudos y los devuelve en un diccionario."""
    data_dir = Path(data_dir)
    try:
        return {
            "inventario": pd.read_csv(data_dir / "inventario_central_v2.csv"),
            "transacciones": pd.read_csv(data_dir / "transacciones_logistica_v2.csv"),
            "feedback": pd.read_csv(data_dir / "feedback_clientes_v2.csv"),
        }
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No encontré los CSV en '{data_dir}'. "
            "Verifica que los tres archivos estén en la carpeta data/."
        ) from exc


# ---------------------------------------------------------------------------
# Health Score
# ---------------------------------------------------------------------------
def _validez(df: pd.DataFrame, reglas: dict) -> float:
    """Proporción de celdas que cumplen las reglas de validez del dataset."""
    if not reglas:
        return 1.0
    total, validas = 0, 0
    for col, regla in reglas.items():
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        total += len(serie)
        validas += regla(serie).sum()
    return validas / total if total else 1.0


def calcular_health_score(df: pd.DataFrame, reglas_validez: dict | None = None,
                          subset_dupes: list | None = None) -> dict:
    """
    Health Score = promedio ponderado de tres dimensiones de calidad:
    completitud (celdas no nulas), unicidad (filas no duplicadas) y
    validez (valores dentro de rangos lógicos de negocio).
    """
    completitud = 1 - df.isna().mean().mean()
    unicidad = 1 - df.duplicated(subset=subset_dupes).mean()
    validez = _validez(df, reglas_validez or {})
    score = round((0.4 * completitud + 0.2 * unicidad + 0.4 * validez) * 100, 1)
    return {
        "score": score,
        "completitud_%": round(completitud * 100, 1),
        "unicidad_%": round(unicidad * 100, 1),
        "validez_%": round(validez * 100, 1),
    }


REGLAS_INVENTARIO = {
    "Stock_Actual": lambda s: s >= 0,
    "Costo_Unitario_USD": lambda s: (s >= 1) & (s <= 10_000),
    "Lead_Time_Dias": lambda s: pd.to_numeric(s, errors="coerce").between(0, 90),
}

REGLAS_TRANSACCIONES = {
    "Cantidad_Vendida": lambda s: s > 0,
    "Tiempo_Entrega_Real": lambda s: (s > 0) & (s < EXTREMO_TIEMPO_ENTREGA),
}

REGLAS_FEEDBACK = {
    "Rating_Producto": lambda s: s.between(1, 5),
    "Edad_Cliente": lambda s: s.between(18, EDAD_MAXIMA),
}


def reporte_nulidad(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de nulos por columna, ordenado de peor a mejor."""
    pct = (df.isna().mean() * 100).round(2)
    return (pct[pct > 0].sort_values(ascending=False)
            .rename("% Nulos").reset_index()
            .rename(columns={"index": "Columna"}))


# ---------------------------------------------------------------------------
# Limpieza: Inventario
# ---------------------------------------------------------------------------
def _parsear_lead_time(valor) -> float:
    """Convierte los formatos mixtos de lead time a días numéricos."""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if texto.lower() == "inmediato":
        return 0.0
    if "-" in texto:  # rangos tipo "25-30 días" -> punto medio
        numeros = [int(n) for n in pd.Series([texto]).str.findall(r"\d+")[0]]
        return float(np.mean(numeros)) if numeros else np.nan
    try:
        return float(texto)
    except ValueError:
        return np.nan


MAPA_CATEGORIAS = {
    "smart-phone": "Smartphones", "Smartphones": "Smartphones",
    "LAPTOP": "Laptops", "Laptops": "Laptops",
    "Accesorios": "Accesorios", "Monitores": "Monitores",
    "Tablets": "Tablets", "???": "Sin Categoría",
}

MAPA_BODEGAS = {
    "norte": "Norte", "Norte": "Norte", "Sur": "Sur",
    "Occidente": "Occidente", "ZONA_FRANCA": "Zona Franca",
    "BOD-EXT-99": "Externa (BOD-EXT-99)",
}


def limpiar_inventario(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Limpia el maestro de productos y documenta cada decisión."""
    df = df.copy()
    log = []

    # 1. Etiquetas inconsistentes: mismo concepto escrito de varias formas
    df["Categoria"] = df["Categoria"].map(MAPA_CATEGORIAS).fillna("Sin Categoría")
    df["Bodega_Origen"] = df["Bodega_Origen"].map(MAPA_BODEGAS).fillna(df["Bodega_Origen"])
    log.append({"accion": "Estandarizar Categoria y Bodega_Origen",
                "filas": len(df),
                "justificacion": "Un mismo concepto aparecía con varias grafías "
                                 "(smart-phone/Smartphones, norte/Norte); sin unificar, "
                                 "cualquier agrupación queda fragmentada."})

    # 2. Stock negativo: contablemente imposible -> se lleva a 0 y se marca
    n_neg = int((df["Stock_Actual"] < 0).sum())
    df["Flag_Stock_Negativo"] = df["Stock_Actual"] < 0
    df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = 0
    log.append({"accion": "Stock negativo llevado a 0", "filas": n_neg,
                "justificacion": "Existencias negativas violan la lógica contable; se "
                                 "interpretan como error de conteo y se conservan marcadas "
                                 "en lugar de eliminarse, para no perder el producto."})

    # 3. Stock nulo: imputación con mediana por categoría (distribución sesgada)
    n_null = int(df["Stock_Actual"].isna().sum())
    df["Stock_Actual"] = df.groupby("Categoria")["Stock_Actual"].transform(
        lambda s: s.fillna(s.median()))
    log.append({"accion": "Imputar Stock_Actual nulo con mediana por categoría",
                "filas": n_null,
                "justificacion": "La mediana es robusta a los extremos y respeta el "
                                 "perfil de stock típico de cada categoría."})

    # 4. Costos atípicos ($0.05 y $850,000): fuera de rango comercial creíble
    fuera = ~df["Costo_Unitario_USD"].between(1, 10_000)
    n_out = int(fuera.sum())
    df.loc[fuera, "Costo_Unitario_USD"] = np.nan
    df["Costo_Unitario_USD"] = df.groupby("Categoria")["Costo_Unitario_USD"].transform(
        lambda s: s.fillna(s.median()))
    log.append({"accion": "Costos fuera de [1, 10.000] USD imputados con mediana de categoría",
                "filas": n_out,
                "justificacion": "Un costo de $850k o de $0.05 en retail tecnológico es un "
                                 "error de captura evidente; imputar con la mediana evita "
                                 "distorsionar todo el cálculo de márgenes."})

    # 5. Lead time con formatos mezclados ("25-30 días", "Inmediato", números)
    n_lt_null = int(df["Lead_Time_Dias"].isna().sum())
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].apply(_parsear_lead_time)
    mediana_lt = df["Lead_Time_Dias"].median()
    df["Lead_Time_Dias"] = df["Lead_Time_Dias"].fillna(mediana_lt)
    log.append({"accion": "Normalizar Lead_Time_Dias a días numéricos e imputar nulos",
                "filas": n_lt_null,
                "justificacion": "Los rangos ('25-30 días') se llevan a su punto medio y "
                                 "'Inmediato' a 0; los nulos toman la mediana global "
                                 f"({mediana_lt:.0f} días)."})

    # 6. Fechas a tipo datetime para poder calcular antigüedad de revisión
    df["Ultima_Revision"] = pd.to_datetime(df["Ultima_Revision"], errors="coerce")

    return df, log


# ---------------------------------------------------------------------------
# Limpieza: Transacciones
# ---------------------------------------------------------------------------
MAPA_CIUDADES = {
    "MED": "Medellín", "Medellín": "Medellín",
    "BOG": "Bogotá", "Bogotá": "Bogotá",
    "Ventas_Web": "Venta Web (sin ciudad)",
}


def limpiar_transacciones(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Limpia el histórico de ventas y documenta cada decisión."""
    df = df.copy()
    log = []

    # 1. Fechas: llegan como dd/mm/yyyy en texto
    df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], format="%d/%m/%Y", errors="coerce")

    # 2. Cantidades negativas: se interpretan como devoluciones, no ventas.
    n_neg = int((df["Cantidad_Vendida"] < 0).sum())
    df["Es_Devolucion"] = df["Cantidad_Vendida"] < 0
    df = df[~df["Es_Devolucion"]].copy()
    log.append({"accion": "Excluir transacciones con cantidad negativa", "filas": n_neg,
                "justificacion": "Una cantidad de -5 no es una venta; tratarla como venta "
                                 "distorsiona el ingreso. Representan el 1% de los "
                                 "registros y se excluyen del análisis comercial."})

    # 3. Tiempo de entrega 999: sentinel de 'sin dato', no un envío de 3 años
    n_999 = int((df["Tiempo_Entrega_Real"] == EXTREMO_TIEMPO_ENTREGA).sum())
    df.loc[df["Tiempo_Entrega_Real"] == EXTREMO_TIEMPO_ENTREGA, "Tiempo_Entrega_Real"] = np.nan
    df["Tiempo_Entrega_Real"] = df.groupby("Ciudad_Destino")["Tiempo_Entrega_Real"].transform(
        lambda s: s.fillna(s.median()))
    log.append({"accion": "Reemplazar tiempo de entrega 999 por mediana de su ciudad",
                "filas": n_999,
                "justificacion": "999 es un valor centinela del sistema; la mediana por "
                                 "ciudad conserva las diferencias logísticas regionales."})

    # 4. Costo de envío nulo: mediana por ciudad (el flete depende del destino)
    n_envio = int(df["Costo_Envio"].isna().sum())
    df["Costo_Envio"] = df.groupby("Ciudad_Destino")["Costo_Envio"].transform(
        lambda s: s.fillna(s.median()))
    log.append({"accion": "Imputar Costo_Envio nulo con mediana por ciudad", "filas": n_envio,
                "justificacion": "El flete varía por destino; usar la mediana de la misma "
                                 "ciudad es más fiel que un promedio global."})

    # 5. Estado de envío nulo: no se inventa un estado, se etiqueta como tal
    n_estado = int(df["Estado_Envio"].isna().sum())
    df["Estado_Envio"] = df["Estado_Envio"].fillna("Sin Registro")
    log.append({"accion": "Estado_Envio nulo -> 'Sin Registro'", "filas": n_estado,
                "justificacion": "Imputar un estado (p. ej. la moda 'Entregado') falsearía "
                                 "el seguimiento de última milla; la ausencia es información."})

    # 6. Ciudades con alias (MED/BOG) y el pseudo-destino 'Ventas_Web'
    df["Ciudad_Destino"] = df["Ciudad_Destino"].map(MAPA_CIUDADES).fillna(df["Ciudad_Destino"])
    log.append({"accion": "Estandarizar Ciudad_Destino", "filas": len(df),
                "justificacion": "MED y Medellín eran la misma ciudad partida en dos; "
                                 "'Ventas_Web' no es una ciudad y se etiqueta aparte."})

    return df, log


# ---------------------------------------------------------------------------
# Limpieza: Feedback
# ---------------------------------------------------------------------------
def limpiar_feedback(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Limpia la voz del cliente y documenta cada decisión."""
    df = df.copy()
    log = []

    # 1. Duplicados: varias opiniones apuntando a la misma transacción.
    n_dup = int(df.duplicated(subset="Transaccion_ID").sum())
    df = df.drop_duplicates(subset="Transaccion_ID", keep="first").copy()
    log.append({"accion": "Eliminar feedback duplicado por Transaccion_ID", "filas": n_dup,
                "justificacion": "Una transacción con múltiples registros de feedback "
                                 "duplica su peso en NPS y ratings; se conserva el primero."})

    # 2. Rating 99: sentinel de 'sin calificación' en una escala 1-5
    n_99 = int((df["Rating_Producto"] == EXTREMO_RATING).sum())
    df.loc[df["Rating_Producto"] == EXTREMO_RATING, "Rating_Producto"] = np.nan
    moda = df["Rating_Producto"].mode()[0]
    df["Rating_Producto"] = df["Rating_Producto"].fillna(moda)
    log.append({"accion": f"Rating_Producto 99 imputado con la moda ({moda:.0f})", "filas": n_99,
                "justificacion": "99 no existe en una escala 1-5. Al ser una variable "
                                 "ordinal discreta, la moda es el imputador natural."})

    # 3. Edades imposibles (hasta 195 años): mediana, que no se deja arrastrar
    n_edad = int((df["Edad_Cliente"] > EDAD_MAXIMA).sum())
    df.loc[df["Edad_Cliente"] > EDAD_MAXIMA, "Edad_Cliente"] = np.nan
    df["Edad_Cliente"] = df["Edad_Cliente"].fillna(df["Edad_Cliente"].median())
    log.append({"accion": f"Edades > {EDAD_MAXIMA} imputadas con la mediana", "filas": n_edad,
                "justificacion": "195 años es biológicamente imposible; la mediana evita "
                                 "el sesgo que los outliers imprimen a la media."})

    # 4. Ticket de soporte con doble codificación (Sí/No y 1/0)
    df["Ticket_Soporte_Abierto"] = df["Ticket_Soporte_Abierto"].map(
        {"Sí": "Sí", "1": "Sí", "No": "No", "0": "No"}).fillna("No")
    log.append({"accion": "Unificar Ticket_Soporte_Abierto a Sí/No", "filas": len(df),
                "justificacion": "El sistema mezclaba texto y binarios para el mismo campo."})

    # 5. Recomendación de marca: el nulo se hace explícito
    n_rec = int(df["Recomienda_Marca"].isna().sum())
    df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna("Sin respuesta")
    log.append({"accion": "Recomienda_Marca nulo -> 'Sin respuesta'", "filas": n_rec,
                "justificacion": "No responder es en sí una señal; no se imputa una opinión."})

    # 6. Comentarios placeholder ('---', 'N/A') se tratan como vacíos
    df["Comentario_Texto"] = df["Comentario_Texto"].replace({"---": np.nan, "N/A": np.nan})

    # 7. NPS: se normaliza a escala 0-100 y se clasifica para lectura gerencial
    df["NPS_Normalizado"] = ((df["Satisfaccion_NPS"] + 100) / 2).round(1)
    df["NPS_Categoria"] = pd.cut(df["Satisfaccion_NPS"], bins=[-101, -33, 33, 101],
                                 labels=["Detractor", "Pasivo", "Promotor"])
    log.append({"accion": "Normalizar NPS a 0-100 y clasificar Detractor/Pasivo/Promotor",
                "filas": len(df),
                "justificacion": "La escala original (-100 a 100) confunde en tableros; la "
                                 "clasificación estándar facilita la lectura de lealtad."})

    return df, log


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
def ejecutar_pipeline(data_dir: Path = DATA_DIR) -> dict:
    """
    Corre la auditoría completa: carga, health score 'antes', limpieza,
    health score 'después' y consolidación del log de decisiones.
    """
    crudos = cargar_datos(data_dir)
    reglas = {"inventario": REGLAS_INVENTARIO,
              "transacciones": REGLAS_TRANSACCIONES,
              "feedback": REGLAS_FEEDBACK}
    subset = {"inventario": ["SKU_ID"],
              "transacciones": ["Transaccion_ID"],
              "feedback": ["Transaccion_ID"]}
    limpiadores = {"inventario": limpiar_inventario,
                   "transacciones": limpiar_transacciones,
                   "feedback": limpiar_feedback}

    resultado = {"crudos": crudos, "limpios": {}, "health": {}, "logs": {}, "nulidad": {}}
    for nombre, df in crudos.items():
        antes = calcular_health_score(df, reglas[nombre], subset[nombre])
        df_limpio, log = limpiadores[nombre](df)
        despues = calcular_health_score(df_limpio, reglas[nombre], subset[nombre])
        resultado["limpios"][nombre] = df_limpio
        resultado["health"][nombre] = {"antes": antes, "despues": despues}
        resultado["logs"][nombre] = log
        resultado["nulidad"][nombre] = reporte_nulidad(df)
    return resultado


def log_a_dataframe(logs: dict) -> pd.DataFrame:
    """Convierte los logs de limpieza en una tabla exportable (el 'rastro')."""
    filas = [{"Dataset": nombre, "Acción": e["accion"],
              "Registros afectados": e["filas"], "Justificación": e["justificacion"]}
             for nombre, entradas in logs.items() for e in entradas]
    return pd.DataFrame(filas)
