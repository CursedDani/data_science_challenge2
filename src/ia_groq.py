"""
ia_groq.py — Módulo de recomendaciones estratégicas con Groq (Llama-3).

Recibe el resumen estadístico de los datos que el usuario filtró en la app
y devuelve tres párrafos de recomendación para la junta directiva.
La API key nunca se escribe en el código: se toma de la variable de entorno
GROQ_API_KEY, de st.secrets o del campo de la barra lateral.
"""

import os

MODELO = "llama-3.3-70b-versatile"

PROMPT_SISTEMA = (
    "Eres un consultor senior de datos para TechLogistics S.A.S., un retail "
    "tecnológico. Recibirás un resumen estadístico del negocio ya filtrado "
    "por el usuario. Responde en español con EXACTAMENTE tres párrafos de "
    "recomendación estratégica dirigidos a la junta directiva: (1) rentabilidad "
    "y fuga de capital, (2) logística y experiencia del cliente, (3) control de "
    "inventario y riesgo operativo. Sé concreto, cita las cifras del resumen y "
    "propone acciones accionables. Sin listas ni encabezados, solo prosa."
)


def obtener_api_key(key_manual: str | None = None) -> str | None:
    """Busca la API key en orden: campo de la app > st.secrets > entorno."""
    if key_manual:
        return key_manual.strip()
    try:  # st.secrets lanza excepción si no hay archivo secrets.toml
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def generar_recomendaciones(resumen_estadistico: str,
                            api_key: str | None = None) -> str:
    """
    Envía el resumen a Llama-3 vía Groq y devuelve las recomendaciones.
    Cualquier fallo (sin key, sin red, cuota) regresa un mensaje amable
    en lugar de romper la app.
    """
    key = obtener_api_key(api_key)
    if not key:
        return ("No hay una API key de Groq configurada. Ingrésala en la "
                "barra lateral o define la variable de entorno GROQ_API_KEY. "
                "Puedes crear una gratis en https://console.groq.com/keys")

    try:
        from groq import Groq
        cliente = Groq(api_key=key)
        respuesta = cliente.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user",
                 "content": f"Resumen estadístico del negocio:\n{resumen_estadistico}"},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        return respuesta.choices[0].message.content
    except ImportError:
        return "Falta instalar la librería `groq` (pip install groq)."
    except Exception as exc:
        return f"No pude obtener la recomendación de Groq: {exc}"
