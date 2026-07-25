"""
main_app.py — Dashboard DSS de TechLogistics S.A.S.

Capa de presentación en Streamlit. Toda la lógica de limpieza vive en
src/limpieza.py, el análisis en src/eda.py y la IA en src/ia_groq.py.

Ejecutar con:  streamlit run main_app.py
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import eda, ia_groq, limpieza

# ---------------------------------------------------------------------------
# Configuración general y estilo
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechLogistics DSS",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETA = px.colors.qualitative.Set2
COLOR_MALO, COLOR_BUENO = "#e74c3c", "#2ecc71"

st.markdown("""
    <style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1a5276; margin-bottom: 0; }
    .subtitle   { color: #7f8c8d; margin-top: 0.2rem; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 12px; padding: 12px 16px;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos (cacheados: la limpieza corre una sola vez por sesión)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Auditando y limpiando los tres sistemas...")
def preparar_datos():
    resultado = limpieza.ejecutar_pipeline()
    maestro = eda.construir_fuente_verdad(
        resultado["limpios"]["inventario"],
        resultado["limpios"]["transacciones"],
        resultado["limpios"]["feedback"],
    )
    return resultado, maestro


try:
    resultado, maestro = preparar_datos()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: filtros globales + configuración de IA
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📦 TechLogistics DSS")
    st.caption("Sistema de Soporte a la Decisión · Challenge 02")

    st.header("🎛️ Filtros")
    categorias = st.multiselect(
        "Categoría", sorted(maestro["Categoria"].dropna().unique()),
        placeholder="Todas")
    ciudades = st.multiselect(
        "Ciudad destino", sorted(maestro["Ciudad_Destino"].dropna().unique()),
        placeholder="Todas")
    canales = st.multiselect(
        "Canal de venta", sorted(maestro["Canal_Venta"].dropna().unique()),
        placeholder="Todos")
    fecha_min = maestro["Fecha_Venta"].min().date()
    fecha_max = maestro["Fecha_Venta"].max().date()
    rango = st.date_input("Rango de fechas", (fecha_min, fecha_max),
                          min_value=fecha_min, max_value=fecha_max)

    st.divider()
    st.header("🤖 IA (Groq)")
    api_key_manual = st.text_input("API key de Groq", type="password",
                                   help="Opcional si definiste GROQ_API_KEY "
                                        "en el entorno o en st.secrets")

# Aplicación de los filtros sobre la fuente de verdad
df = maestro.copy()
if categorias:
    df = df[df["Categoria"].isin(categorias)]
if ciudades:
    df = df[df["Ciudad_Destino"].isin(ciudades)]
if canales:
    df = df[df["Canal_Venta"].isin(canales)]
if isinstance(rango, tuple) and len(rango) == 2:
    df = df[(df["Fecha_Venta"].dt.date >= rango[0])
            & (df["Fecha_Venta"].dt.date <= rango[1])]

if df.empty:
    st.warning("Los filtros seleccionados no dejan ningún registro. "
               "Amplía la selección para continuar.")
    st.stop()

# ---------------------------------------------------------------------------
# Encabezado + KPIs
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">Del Caos al Storytelling 📊</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Auditoría, integración y estrategia sobre los '
            'tres sistemas de TechLogistics S.A.S.</p>', unsafe_allow_html=True)

k = eda.kpis_generales(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ingreso total", f"${k['ingreso_total']/1e6:,.2f}M")
c2.metric("Margen (catalogado)", f"${k['margen_total']/1e6:,.2f}M",
          f"{k['margen_pct']:.1f}%")
c3.metric("Transacciones", f"{k['transacciones']:,}")
c4.metric("Ventas fantasma", f"{k['pct_fantasma']:.1f}%",
          "-riesgo de catálogo", delta_color="inverse")
c5.metric("NPS promedio", f"{k['nps_promedio']:.1f}")

tab_audit, tab_preguntas, tab_ia, tab_descargas = st.tabs([
    "🧹 Auditoría de Calidad", "🔎 Las 5 Preguntas", "🤖 Recomendación IA", "⬇️ Descargas"])

# ---------------------------------------------------------------------------
# TAB 1 — Auditoría de calidad (health score antes/después + log)
# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Health Score por dataset: antes vs. después")
    st.caption("Score = 40% completitud + 20% unicidad + 40% validez de negocio.")

    cols = st.columns(3)
    for col, (nombre, h) in zip(cols, resultado["health"].items()):
        with col:
            delta = round(h["despues"]["score"] - h["antes"]["score"], 1)
            st.metric(nombre.capitalize(), f"{h['despues']['score']}/100",
                      f"+{delta} pts tras limpieza")
            fig = go.Figure()
            for etapa, color in [("antes", COLOR_MALO), ("despues", COLOR_BUENO)]:
                fig.add_trace(go.Bar(
                    name=etapa.capitalize(),
                    x=["Completitud", "Unicidad", "Validez"],
                    y=[h[etapa]["completitud_%"], h[etapa]["unicidad_%"],
                       h[etapa]["validez_%"]],
                    marker_color=color))
            fig.update_layout(barmode="group", height=260, showlegend=True,
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_range=[0, 105])
            st.plotly_chart(fig, use_container_width=True,
                            key=f"health_{nombre}")

    st.subheader("Nulidad por columna (datos crudos)")
    cols_nul = st.columns(3)
    for col, (nombre, tabla) in zip(cols_nul, resultado["nulidad"].items()):
        with col:
            st.markdown(f"**{nombre.capitalize()}**")
            if tabla.empty:
                st.success("Sin nulos 🎉")
            else:
                st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.subheader("Bitácora de decisiones (el rastro del consultor)")
    st.caption("Qué se eliminó, qué se imputó y por qué — nada quedó sin justificar.")
    st.dataframe(limpieza.log_a_dataframe(resultado["logs"]),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2 — Las 5 preguntas de alta gerencia
# ---------------------------------------------------------------------------
with tab_preguntas:

    # ---- P1: margen negativo -------------------------------------------------
    st.subheader("1️⃣ Fuga de capital: SKUs vendidos con margen negativo")
    negativos = eda.skus_margen_negativo(df)
    perdida_total = negativos["Margen_Total_USD"].sum()
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig = px.bar(negativos.head(15), x="Margen_Total_USD", y="SKU_ID",
                     color="Categoria", orientation="h",
                     color_discrete_sequence=PALETA,
                     labels={"Margen_Total_USD": "Margen total (USD)"},
                     title="Top 15 SKUs que destruyen valor")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.metric("SKUs con margen negativo", len(negativos))
        st.metric("Pérdida acumulada", f"${perdida_total:,.0f}")
        canal = eda.margen_por_canal(df)
        if not canal.empty:
            fig = px.pie(canal, values=canal["Perdida_USD"].abs(),
                         names="Canal_Venta", title="Pérdida por canal",
                         color_discrete_sequence=PALETA, hole=0.45)
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
    st.info("**Lectura:** la pérdida no se concentra exclusivamente en Online: "
            "se reparte entre canales, lo que apunta a una falla de *pricing* "
            "por producto (costo mal cargado o descuento excesivo) más que a "
            "una guerra de precios de un solo canal.")

    st.divider()

    # ---- P2: crisis logística --------------------------------------------------
    st.subheader("2️⃣ Crisis logística: ¿dónde castiga más cada día de retraso?")
    dim = st.radio("Analizar por", ["Ciudad_Destino", "Bodega_Origen"],
                   horizontal=True, format_func=lambda x: x.replace("_", " "))
    corr = eda.correlacion_entrega_nps(df, por=dim)
    if corr.empty:
        st.warning("No hay muestras suficientes con el filtro actual.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(corr, x="Correlacion", y=dim, orientation="h",
                         color="Correlacion", color_continuous_scale="RdYlGn",
                         title="Correlación Tiempo de Entrega ↔ NPS")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.scatter(corr, x="Entrega_Promedio_Dias", y="NPS_Promedio",
                             size="Muestras", text=dim,
                             color="Correlacion", color_continuous_scale="RdYlGn",
                             title="Entrega promedio vs NPS promedio",
                             labels={"Entrega_Promedio_Dias": "Días de entrega (prom.)"})
            fig.update_traces(textposition="top center")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        peor = corr.iloc[0]
        st.error(f"**Zona crítica: {peor[dim]}** — correlación de "
                 f"{peor['Correlacion']:.2f} entre días de entrega y NPS, con "
                 f"{peor['Entrega_Promedio_Dias']:.0f} días promedio. Es la primera "
                 "candidata a cambio de operador logístico.")

    st.divider()

    # ---- P3: venta invisible -----------------------------------------------------
    st.subheader("3️⃣ La venta invisible: SKUs fantasma")
    fantasma = eda.impacto_ventas_fantasma(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ingreso en riesgo", f"${fantasma['ingreso_fantasma_usd']:,.0f}",
              f"{fantasma['pct_ingreso_en_riesgo']:.1f}% del total",
              delta_color="inverse")
    c2.metric("Transacciones sin respaldo", f"{fantasma['transacciones_fantasma']:,}")
    c3.metric("SKUs no catalogados", fantasma["skus_fantasma_unicos"])
    col_a, col_b = st.columns(2)
    with col_a:
        serie = (df.assign(Tipo=df["Es_SKU_Fantasma"].map(
                    {True: "Fantasma", False: "Catalogada"}))
                 .groupby([df["Fecha_Venta"].dt.to_period("M").dt.to_timestamp(), "Tipo"])
                 ["Ingreso_USD"].sum().reset_index())
        fig = px.area(serie, x="Fecha_Venta", y="Ingreso_USD", color="Tipo",
                      color_discrete_map={"Fantasma": COLOR_MALO,
                                          "Catalogada": "#3498db"},
                      title="Ingreso mensual: catalogado vs fantasma")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(fantasma["por_canal"], x="Canal_Venta", y="Ingreso_USD",
                     color="Canal_Venta", color_discrete_sequence=PALETA,
                     title="Ingreso fantasma por canal")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.info("**Decisión del consultor:** estas ventas se conservan como ingreso "
            "real (el dinero entró), pero se excluyen del cálculo de margen "
            "porque no existe costo de referencia. El patrón —cientos de SKUs, "
            "todos los canales, todo el periodo— sugiere una falla de "
            "sincronización de catálogo más que fraude puntual.")

    st.divider()

    # ---- P4: paradoja de fidelidad -----------------------------------------------
    st.subheader("4️⃣ Paradoja de fidelidad: mucho stock, clientes molestos")
    paradoja = eda.paradoja_stock_sentimiento(df)
    fig = px.scatter(paradoja, x="Stock_Promedio", y="NPS_Promedio",
                     size="Ventas", color="Rating_Producto", text="Categoria",
                     color_continuous_scale="RdYlGn",
                     title="Disponibilidad vs sentimiento por categoría")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(paradoja.round(2), use_container_width=True, hide_index=True)
    en_paradoja = paradoja[paradoja["Paradoja"]]["Categoria"].tolist()
    if en_paradoja:
        st.warning(f"**Categorías en paradoja:** {', '.join(en_paradoja)}. "
                   "Cuando el rating de producto acompaña al NPS bajo, el problema "
                   "es calidad; si el rating es aceptable pero el margen porcentual "
                   "es alto, el cliente está percibiendo sobrecosto.")

    st.divider()

    # ---- P5: riesgo operativo -----------------------------------------------------
    st.subheader("5️⃣ Riesgo operativo: bodegas que operan a ciegas")
    riesgo = eda.riesgo_operativo_bodegas(df)
    fig = px.scatter(riesgo, x="Dias_Sin_Revision_Prom", y="Tasa_Tickets_Pct",
                     size="Ventas", color="NPS_Promedio", text="Bodega_Origen",
                     color_continuous_scale="RdYlGn",
                     labels={"Dias_Sin_Revision_Prom": "Días sin revisión de stock (prom.)",
                             "Tasa_Tickets_Pct": "Tasa de tickets de soporte (%)"},
                     title="Antigüedad de revisión vs carga de soporte")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(riesgo.round(2), use_container_width=True, hide_index=True)
    if not riesgo.empty:
        ciega = riesgo.iloc[0]
        st.error(f"**Bodega más desactualizada: {ciega['Bodega_Origen']}** — "
                 f"{ciega['Dias_Sin_Revision_Prom']:.0f} días promedio sin conteo "
                 f"físico y {ciega['Tasa_Tickets_Pct']:.1f}% de tickets. Un stock "
                 "que nadie verifica termina prometiendo lo que no puede entregar, "
                 "y eso se paga en soporte y en NPS.")

# ---------------------------------------------------------------------------
# TAB 3 — Recomendación estratégica con IA (Groq / Llama-3)
# ---------------------------------------------------------------------------
with tab_ia:
    st.subheader("Consultor virtual — Llama-3 vía Groq")
    st.caption("El modelo recibe solo el resumen estadístico de los datos que "
               "filtraste (nunca los datos crudos) y devuelve tres párrafos de "
               "recomendación para la junta.")

    resumen = eda.resumen_para_ia(df)
    with st.expander("Ver el resumen estadístico que se envía al modelo"):
        st.code(resumen, language=None)

    if st.button("✨ Generar recomendación estratégica", type="primary"):
        with st.spinner("Consultando a Llama-3..."):
            respuesta = ia_groq.generar_recomendaciones(resumen, api_key_manual)
        st.markdown(respuesta)

# ---------------------------------------------------------------------------
# TAB 4 — Descargas
# ---------------------------------------------------------------------------
with tab_descargas:
    st.subheader("Reportes descargables")

    log_df = limpieza.log_a_dataframe(resultado["logs"])
    st.download_button(
        "📋 Reporte de limpieza (CSV)",
        log_df.to_csv(index=False).encode("utf-8-sig"),
        "reporte_limpieza_techlogistics.csv", "text/csv")

    st.download_button(
        "🗃️ Fuente de verdad filtrada (CSV)",
        df.to_csv(index=False).encode("utf-8-sig"),
        "fuente_verdad_filtrada.csv", "text/csv")

    # Resumen de salud en texto plano, útil para anexar al informe PDF
    lineas = ["REPORTE DE CALIDAD - TECHLOGISTICS S.A.S.", "=" * 45, ""]
    for nombre, h in resultado["health"].items():
        lineas += [f"{nombre.upper()}: {h['antes']['score']} -> "
                   f"{h['despues']['score']} /100"]
    lineas += ["", "Bitácora de decisiones:", ""]
    lineas += [f"- [{r['Dataset']}] {r['Acción']} ({r['Registros afectados']} filas): "
               f"{r['Justificación']}" for _, r in log_df.iterrows()]
    st.download_button(
        "📝 Resumen ejecutivo de calidad (TXT)",
        "\n".join(lineas).encode("utf-8"),
        "resumen_calidad.txt", "text/plain")

st.divider()
st.caption("Challenge 02 · Fundamentos en Ciencia de Datos · EAFIT 2026-1 — "
           "los datos sucios bien interpretados salvan empresas.")
