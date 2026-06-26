# predWC - World Cup 2026 Knockout Predictor

Stacking ensemble model (Random Forest + XGBoost + SVM → Logistic Regression) que predice los ganadores de los 16avos de final del Mundial 2026.

## Requisitos

- Python ≥ 3.12
- curl (para instalar uv)

## Instalación y ejecución

### Opción 1: Auto-instalador (recomendado)

```bash
python install_all.py
```

Esto instala `uv` (si no lo tienes), las dependencias con `uv sync`, y Playwright browsers.

Luego ejecutas el modelo:

```bash
uv run python stacking_model.py
```

### Opción 2: Paso a paso (manual)

**Paso 1 — Instalar uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cierra y abre la terminal, o ejecuta `source ~/.bashrc` / `source ~/.zshrc` para que `uv` esté disponible.

**Paso 2 — Verificar Python 3.12**

```bash
python3 --version
```

Debe mostrar `Python 3.12.x`. Si no lo tienes, instálalo con tu gestor de paquetes o desde [python.org](https://python.org).

**Paso 3 — Instalar dependencias**

```bash
uv sync
```

Esto lee `pyproject.toml` y crea un entorno virtual (`.venv/`) con todas las librerías necesarias: polars, scikit-learn, xgboost, requests, playwright, etc.

**Paso 4 — Instalar navegadores Playwright** (solo si vas a actualizar datos)

```bash
uv run playwright install chromium
```

El modelo en sí no lo necesita, pero `scripts/update_data.py` sí.

**Paso 5 — Ejecutar el modelo**

```bash
uv run python stacking_model.py
```

El modelo:
1. Descarga `results.csv` desde GitHub (~50k partidos internacionales)
2. Carga rankings ELO y ELO histórico
3. Construye 25 features por cada uno de los ~8160 partidos (2018–2026)
4. Entrena 3 modelos base (Random Forest, XGBoost, SVM) con split temporal
5. Entrena meta-modelo (Logistic Regression) con predicciones out-of-sample
6. Evalúa en validación temporal (~1630 partidos más recientes)
7. Re-entrena con todos los datos y predice los 16 cruces de 16avos
8. Simula scores exactos vía Poisson + Monte Carlo (ELO-based)
9. Guarda resultados en `data/knockout_predictions.csv`

**Paso 6 — Ver resultados gráficos**

```bash
uv run python show_results.py
```

Abre 4 ventanas interactivas: avance por partido, probabilidades local/empate/visitante, confianza del modelo, y scores esperados Poisson.

**Paso 7 — (Opcional) Actualizar datasets**

Si quieres regenerar los datos desde cero:

```bash
uv run python scripts/update_data.py
```

Esto actualiza rankings ELO, historial ELO y cruces de knockout desde sus fuentes originales. Requiere Playwright instalado (paso 4).

## Archivos incluidos

| Archivo | Descripción |
|---|---|
| `stacking_model.py` | Modelo stacking: entrena con ~8160 partidos (2018–2026) y predice los 16 cruces |
| `show_results.py` | Visualización interactiva (matplotlib) de resultados |
| `data/` | Datasets pre-generados (results, rankings, ELO history, knockout matches) |
| `pyproject.toml` | Dependencias del proyecto |
| `install_all.py` | Script que instala uv y dependencias automáticamente |

## Salida del modelo

El modelo imprime en consola:
- Accuracy y log-loss sobre validación temporal (~1630 partidos)
- Probabilidad de victoria local/empate/visitante para cada uno de los 16 partidos
- λ esperados (Poisson) y top 5 scores exactos más probables
- Probabilidad de avance (empate se reparte 50/50)

Además guarda `data/knockout_predictions.csv` con todas las predicciones.

## Estructura del proyecto

```
predWC/
├── stacking_model.py        # Modelo principal
├── show_results.py          # Visualización interactiva
├── install_all.py           # Auto-instalador
├── pyproject.toml           # Dependencias
├── data/
│   ├── results.csv           # Partidos históricos (49k)
│   ├── elo_rankings.json     # ELO snapshot
│   ├── elo_history.parquet   # ELO histórico por partido
│   ├── training_dataset.parquet/csv  # Dataset de entrenamiento
│   ├── team_features.parquet/csv     # Features por equipo
│   ├── knockout_matches.json  # Cruces de 16avos
│   └── knockout_predictions.csv      # Predicciones generadas
└── README.md
```

## Notas

- `MAX_DATE` se calcula automáticamente como `hoy - 1 día` — `results.csv` se actualiza a diario con datos hasta el día anterior
- Los datos en `data/` están pre-generados; para actualizarlos se necesita la carpeta `scripts/` (no incluida aquí)
- Las predicciones dividen el empate 50/50 para calcular probabilidad de avance
