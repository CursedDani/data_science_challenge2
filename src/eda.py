"""
eda.py — Integración, feature engineering y análisis de TechLogistics.

Aquí vive la "Sola Fuente de Verdad": el merge de los tres sistemas,
las variables derivadas y una función por cada una de las 5 preguntas
de alta gerencia. main_app.py solo consume estas funciones y las grafica.
"""

import numpy as np
import pandas as pd

FECHA_REFERENCIA = pd.Timestamp("2026-02-28")


# ---------------------------------------------------------------------------
# Integración (Merge)
# ---------------------------------------------------------------------------
def construir_fuente_verdad(inventario: pd.DataFrame,
                            transacciones: pd.DataFrame,
                            feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Une los tres sistemas en una sola tabla a nivel de transacción.

    Decisión sobre el SKU Fantasma: las ventas cuyo SKU no existe en el
    maestro NO se eliminan. Son ingresos reales que la empresa cobró; el
    problema es de catálogo, no de la venta. Se marcan con `Es_SKU_Fantasma`
    para poder cuantificar su impacto, pero quedan excluidas
    del cálculo de margen porque sin costo unitario el margen sería inventado.
    """
    df = transacciones.merge(inventario, on="SKU_ID", how="left", indicator=True)
    df["Es_SKU_Fantasma"] = df["_merge"] == "left_only"
    df = df.drop(columns="_merge")

    df = df.merge(feedback, on="Transaccion_ID", how="left")

    # --- Variables derivadas -------------------------------------------------
    # Ingreso bruto
    df["Ingreso_USD"] = df["Precio_Venta_Final"] * df["Cantidad_Vendida"]

    # Margen de utilidad (tomando en cuenta casos con coston real)
    df["Margen_USD"] = np.where(
        df["Es_SKU_Fantasma"], np.nan,
        (df["Precio_Venta_Final"] - df["Costo_Unitario_USD"]) * df["Cantidad_Vendida"]
        - df["Costo_Envio"])
    df["Margen_Pct"] = (df["Margen_USD"] / df["Ingreso_USD"] * 100).round(2)

    # Brecha de entrega
    df["Brecha_Entrega_Dias"] = df["Tiempo_Entrega_Real"] - df["Lead_Time_Dias"]

    # Antigüedad del último conteo físico del stock
    df["Dias_Sin_Revision"] = (FECHA_REFERENCIA - df["Ultima_Revision"]).dt.days

    # Flag binario de ticket para poder promediar tasas de soporte
    df["Tiene_Ticket"] = (df["Ticket_Soporte_Abierto"] == "Sí").astype("float")
    df.loc[df["Ticket_Soporte_Abierto"].isna(), "Tiene_Ticket"] = np.nan

    return df


def kpis_generales(df: pd.DataFrame) -> dict:
    """KPIs de cabecera para el dashboard."""
    con_margen = df.dropna(subset=["Margen_USD"])
    return {
        "ingreso_total": df["Ingreso_USD"].sum(),
        "margen_total": con_margen["Margen_USD"].sum(),
        "margen_pct": (con_margen["Margen_USD"].sum()
                       / con_margen["Ingreso_USD"].sum() * 100) if len(con_margen) else 0,
        "transacciones": len(df),
        "pct_fantasma": df["Es_SKU_Fantasma"].mean() * 100,
        "entrega_promedio": df["Tiempo_Entrega_Real"].mean(),
        "nps_promedio": df["Satisfaccion_NPS"].mean(),
    }


# ---------------------------------------------------------------------------
# Pregunta 1 — Fuga de capital: SKUs con margen negativo
# ---------------------------------------------------------------------------
def skus_margen_negativo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por SKU y devuelve los que destruyen valor, ordenados por pérdida."""
    por_sku = (df.dropna(subset=["Margen_USD"])
               .groupby(["SKU_ID", "Categoria"], as_index=False)
               .agg(Margen_Total_USD=("Margen_USD", "sum"),
                    Ingreso_Total_USD=("Ingreso_USD", "sum"),
                    Unidades=("Cantidad_Vendida", "sum"),
                    Transacciones=("Transaccion_ID", "count")))
    negativos = por_sku[por_sku["Margen_Total_USD"] < 0].copy()
    return negativos.sort_values("Margen_Total_USD")


def margen_por_canal(df: pd.DataFrame) -> pd.DataFrame:
    """Compara la pérdida por canal para responder si el problema es Online."""
    perdidas = df[(df["Margen_USD"] < 0)]
    return (perdidas.groupby("Canal_Venta", as_index=False)
            .agg(Perdida_USD=("Margen_USD", "sum"),
                 Transacciones=("Transaccion_ID", "count"))
            .sort_values("Perdida_USD"))


# ---------------------------------------------------------------------------
# Pregunta 2 — Crisis logística: entrega vs NPS por ciudad y bodega
# ---------------------------------------------------------------------------
def correlacion_entrega_nps(df: pd.DataFrame, por: str = "Ciudad_Destino") -> pd.DataFrame:
    """
    Correlación (Pearson) entre tiempo de entrega y NPS dentro de cada zona.
    Una correlación muy negativa = cada día extra de entrega castiga la lealtad.
    """
    base = df.dropna(subset=["Satisfaccion_NPS", "Tiempo_Entrega_Real"])
    filas = []
    for zona, grupo in base.groupby(por, observed=True):
        if len(grupo) < 30: # porque las muestras pequeñas nos darían error
            continue
        filas.append({
            por: zona,
            "Correlacion": grupo["Tiempo_Entrega_Real"].corr(grupo["Satisfaccion_NPS"]),
            "Entrega_Promedio_Dias": grupo["Tiempo_Entrega_Real"].mean(),
            "NPS_Promedio": grupo["Satisfaccion_NPS"].mean(),
            "Muestras": len(grupo),
        })
    return (pd.DataFrame(filas).sort_values("Correlacion")
            if filas else pd.DataFrame())


# ---------------------------------------------------------------------------
# Pregunta 3 — La venta invisible (SKUs fantasma)
# ---------------------------------------------------------------------------
def impacto_ventas_fantasma(df: pd.DataFrame) -> dict:
    """Cuantifica en USD el ingreso que fluye sin respaldo de inventario."""
    fantasma = df[df["Es_SKU_Fantasma"]]
    ingreso_total = df["Ingreso_USD"].sum()
    ingreso_fantasma = fantasma["Ingreso_USD"].sum()
    return {
        "ingreso_fantasma_usd": ingreso_fantasma,
        "pct_ingreso_en_riesgo": (ingreso_fantasma / ingreso_total * 100) if ingreso_total else 0,
        "transacciones_fantasma": len(fantasma),
        "skus_fantasma_unicos": fantasma["SKU_ID"].nunique(),
        "por_canal": (fantasma.groupby("Canal_Venta", as_index=False)
                      .agg(Ingreso_USD=("Ingreso_USD", "sum"),
                           Transacciones=("Transaccion_ID", "count"))),
    }


# ---------------------------------------------------------------------------
# Pregunta 4 — Paradoja de fidelidad: stock alto, sentimiento bajo
# ---------------------------------------------------------------------------
def paradoja_stock_sentimiento(df: pd.DataFrame) -> pd.DataFrame:
    base = df[~df["Es_SKU_Fantasma"]]
    tabla = (base.groupby("Categoria", as_index=False, observed=True)
             .agg(Stock_Promedio=("Stock_Actual", "mean"),
                  Rating_Producto=("Rating_Producto", "mean"),
                  NPS_Promedio=("Satisfaccion_NPS", "mean"),
                  Margen_Pct_Promedio=("Margen_Pct", "mean"),
                  Precio_Promedio=("Precio_Venta_Final", "mean"),
                  Costo_Promedio=("Costo_Unitario_USD", "mean"),
                  Ventas=("Transaccion_ID", "count")))
    tabla["Paradoja"] = ((tabla["Stock_Promedio"] > tabla["Stock_Promedio"].median())
                         & (tabla["NPS_Promedio"] < 0))
    return tabla.sort_values("NPS_Promedio")


# ---------------------------------------------------------------------------
# Pregunta 5 — Riesgo operativo: revisiones viejas vs tickets de soporte
# ---------------------------------------------------------------------------
def riesgo_operativo_bodegas(df: pd.DataFrame) -> pd.DataFrame:
    """Relaciona la antigüedad del último conteo con la tasa de tickets."""
    base = df[~df["Es_SKU_Fantasma"]].dropna(subset=["Dias_Sin_Revision"])
    return (base.groupby("Bodega_Origen", as_index=False, observed=True)
            .agg(Dias_Sin_Revision_Prom=("Dias_Sin_Revision", "mean"),
                 Tasa_Tickets_Pct=("Tiene_Ticket", lambda s: s.mean() * 100),
                 NPS_Promedio=("Satisfaccion_NPS", "mean"),
                 Ventas=("Transaccion_ID", "count"))
            .sort_values("Dias_Sin_Revision_Prom", ascending=False))


# ---------------------------------------------------------------------------
# Utilidades para la app
# ---------------------------------------------------------------------------
def resumen_para_ia(df: pd.DataFrame) -> str:
    """
    Condensa el estado del negocio filtrado en un texto corto que el modelo
    de lenguaje puede digerir (no se le envían datos crudos ni PII).
    """
    k = kpis_generales(df)
    fantasma = impacto_ventas_fantasma(df)
    peores = skus_margen_negativo(df).head(5)
    corr = correlacion_entrega_nps(df)

    lineas = [
        f"Transacciones analizadas: {k['transacciones']:,}",
        f"Ingreso total: ${k['ingreso_total']:,.0f} USD",
        f"Margen total (SKUs catalogados): ${k['margen_total']:,.0f} USD ({k['margen_pct']:.1f}%)",
        f"Ventas con SKU fantasma: {k['pct_fantasma']:.1f}% de las transacciones, "
        f"${fantasma['ingreso_fantasma_usd']:,.0f} USD en riesgo "
        f"({fantasma['pct_ingreso_en_riesgo']:.1f}% del ingreso)",
        f"Tiempo de entrega promedio: {k['entrega_promedio']:.1f} días",
        f"NPS promedio: {k['nps_promedio']:.1f} (escala -100 a 100)",
    ]
    if not peores.empty:
        top = ", ".join(f"{r.SKU_ID} (${r.Margen_Total_USD:,.0f})"
                        for r in peores.itertuples())
        lineas.append(f"Top SKUs con margen negativo: {top}")
    if not corr.empty:
        peor = corr.iloc[0]
        lineas.append(f"Zona con peor relación entrega-NPS: {peor.iloc[0]} "
                      f"(corr={peor['Correlacion']:.2f}, "
                      f"{peor['Entrega_Promedio_Dias']:.0f} días promedio)")
    return "\n".join(lineas)
