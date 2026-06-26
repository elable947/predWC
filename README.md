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

**Paso 7 — (Opcional) Predicciones con NLP**

El proyecto incluye una versión extendida que usa 24 features derivadas de NLP (news embeddings + YouTube sentiment). Para ejecutarla:

```bash
uv run python stacking_model_nlp.py
```

Para visualizar los resultados NLP:

```bash
uv run python show_results.py --nlp
```

**Paso 8 — (Opcional) Actualizar datasets**

Si quieres regenerar los datos desde cero:

```bash
uv run python scripts/update_data.py
```

Esto actualiza rankings ELO, historial ELO y cruces de knockout desde sus fuentes originales. Requiere Playwright instalado (paso 4).

## Archivos incluidos

| Archivo | Descripción |
|---|---|---|
| `stacking_model.py` | Modelo stacking base (25 features) — entrena y predice los 16 cruces |
| `stacking_model_nlp.py` | Modelo stacking + 24 features NLP (news embeddings + YouTube sentiment) |
| `show_results.py` | Visualización interactiva (matplotlib); usar `--nlp` para versión NLP |
| `data/` | Datasets pre-generados (ELO, knockout matches, NLP features, raw data) |
| `pyproject.toml` | Dependencias del proyecto |
| `install_all.py` | Auto-instalador: instala uv, dependencias y Playwright |

## Salida del modelo

Ambos modelos imprimen en consola:
- Accuracy y log-loss sobre validación temporal (~1630 partidos, split 80/20 por fecha)
- Probabilidad de victoria local / empate / visitante para cada uno de los 16 partidos
- λ esperados (Poisson) y top 5 scores exactos más probables
- Probabilidad de avance (empate se reparte 50/50)

Además guardan:
- `data/knockout_predictions.csv` — predicciones del modelo base
- `data/knockout_predictions_nlp.csv` — predicciones del modelo NLP

## Estructura del proyecto

```
predWC/
├── stacking_model.py              # Modelo base
├── stacking_model_nlp.py          # Modelo con NLP
├── show_results.py                # Visualización
├── install_all.py                 # Auto-instalador
├── pyproject.toml                 # Dependencias
├── README.md
├── data/
│   ├── elo_rankings.json          # ELO snapshot (eloratings.net)
│   ├── elo_history.parquet        # ELO histórico por partido
│   ├── knockout_matches.json      # 16 cruces de 16avos
│   ├── knockout_predictions.csv   # Predicciones base
│   ├── knockout_predictions_nlp.csv  # Predicciones NLP
│   ├── team_nlp_features.json     # 11 PC de news + YouTube sentiment por equipo
│   ├── raw_news/                  # Artículos Wikipedia + BBC/ESPN por equipo
│   └── raw_comments/              # Comentarios de YouTube por equipo
└── scripts/ (no incluido en git)
    ├── fetch_elo.py, fetch_elo_history.py, fetch_knockout_matches.py
    ├── fetch_team_news.py, fetch_team_comments.py, compute_team_embeddings.py
    └── build_dataset.py, update_data.py
```

## Notas

- `MAX_DATE` se calcula como `hoy - 1 día` — los datos de entrenamiento se actualizan automáticamente
- Los datos en `data/` están pre-generados; para regenerarlos se necesita la carpeta `scripts/` (ver sección de actualización)
- Las predicciones de avance dividen el empate 50/50 entre local y visitante
- El modelo NLP requiere datos generados por los scripts de scraping (news + YouTube); si faltan, el script mostrará un error con instrucciones
