# TechLogistics DSS — Challenge 02

Dashboard de soporte a la decisión que audita, limpia e integra los tres
sistemas de TechLogistics S.A.S. (inventario, logística y feedback) y responde
las 5 preguntas de alta gerencia con evidencia visual + recomendaciones de IA.

**Integrantes del equipo:**

| Nombre completo | Cédula         |
| --------------- | -------------- |
| Daniel Felipe Arango Guarín | 1018227831 |
| Daniel Correa Botero | 1023624609 |
| Miguel Ángel Cano Salinas | 1023522662 |

## Enlace del Dashboard: https://datasciencechallenge2-nakcyclge3asonmnaojpg9.streamlit.app

## Estructura

```
techlogistics_dss/
├── main_app.py          # App de Streamlit (solo presentación)
├── src/
│   ├── limpieza.py      # Auditoría, health score y limpieza con bitácora
│   ├── eda.py           # Merge, feature engineering y las 5 preguntas
│   └── ia_groq.py       # Recomendaciones con Llama-3 vía Groq
├── data/                # Los tres CSV crudos
└── requirements.txt
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
streamlit run main_app.py
```

Para el módulo de IA necesitas una API key gratuita de [Groq](https://console.groq.com/keys).
Puedes pegarla en la barra lateral de la app, o dejarla configurada:

```bash
export GROQ_API_KEY="tu_key"          # Linux / Mac
setx GROQ_API_KEY "tu_key"            # Windows
```

(también funciona vía `.streamlit/secrets.toml` con `GROQ_API_KEY = "tu_key"`).

## Decisiones de limpieza (resumen)

Cada transformación queda registrada en una bitácora descargable desde la app
(pestaña Descargas). Las más relevantes: los costos fuera de $1–$10.000 y las
edades mayores a 100 años se imputan con la **mediana** (robusta a outliers);
el rating 99 se imputa con la **moda** (variable ordinal); los estados de envío
nulos se etiquetan como "Sin Registro" en vez de imputarse, porque la ausencia
es información. Las ventas con SKU fantasma **no se eliminan**: son ingreso
real, se marcan y se excluyen solo del cálculo de margen.

## Health Score

`40% completitud + 20% unicidad + 40% validez de negocio`, calculado antes y
después de la limpieza para cada dataset (visible en la pestaña Auditoría).
