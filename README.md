# predWC — World Cup 2026 Knockout Predictor

Stacking ensemble (Random Forest + XGBoost + SVM → Logistic Regression) que predice los 16avos de final del Mundial 2026. Versión con features NLP opcionales (news embeddings + YouTube sentiment).

## Requisitos

- **Python ≥ 3.12** (verificar con `python3 --version`)
- **curl** (para instalar uv)

Probado en **Manjaro** y **Ubuntu**.

## Instalación

### Opción recomendada: auto-instalador

```bash
python install_all.py
```

Esto hace todo automáticamente:
1. Instala `uv` si no lo tienes
2. Crea entorno virtual e instala todas las dependencias con `uv sync`
3. Instala Playwright + Chromium (necesario para actualizar datos)

### Opción manual

```bash
# 1. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Cierra y abre la terminal, o: source ~/.bashrc

# 2. Verificar Python 3.12
python3 --version

# 3. Instalar dependencias
uv sync

# 4. Instalar Chromium para Playwright
uv run playwright install --with-deps chromium
```

## Uso

### Modelo base (sin NLP)

```bash
uv run python stacking_model.py
```

Construye 25 features por partido (ELO, forma reciente, h2h, peso del torneo), entrena stacking con split temporal 80/20, evalúa en ~1634 partidos futuros (sin leakage) y predice los 16 cruces con scores Poisson + Monte Carlo.

### Modelo con NLP

```bash
uv run python stacking_model.py --nlp
```

Agrega 24 features NLP (11 componentes PCA de news embeddings + YouTube sentiment por equipo). Requiere que exista `data/team_nlp_features.json` (generado con los scripts de NLP).

### Ver resultados gráficos

```bash
uv run python show_results.py            # resultados base
uv run python show_results.py --nlp      # resultados NLP
uv run python show_results.py --save     # guarda PNG en vez de mostrar ventanas
```

Abre 4 ventanas: avance, probabilidades, confianza y scores Poisson.

### Evaluación extendida del modelo

```bash
uv run python evaluate_model.py
```

Calcula F1 macro/weighted, MCC, Brier score, matriz de confusión, calibración por bins y comparación por modelo base (RF / XGB / SVM / Stacking).

## Actualizar datos

### ELO rankings + historial

```bash
uv run python scripts/update_data.py
```

### Pipeline NLP completo (noticias + YouTube + embeddings)

```bash
# 1. Scrapea noticias (Wikipedia + BBC/ESPN → trafilatura)
uv run python scripts/fetch_team_news.py

# 2. Comentarios YouTube (necesita API key en apis.txt)
uv run python scripts/fetch_team_comments.py

# 3. Recalcula embeddings → PCA → sentiment
uv run python scripts/compute_team_embeddings.py

# 4. Re-entrenar con NLP
uv run python stacking_model.py --nlp
```

La API key de YouTube va en `apis.txt` con este formato:
```
YOUTUBE:
 - AIzaSy...
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `stacking_model.py` | Modelo stacking (25 features base; `--nlp` agrega 24 NLP) |
| `evaluate_model.py` | Evaluación extendida (F1, MCC, Brier, calibración) |
| `show_results.py` | Visualización interactiva |
| `install_all.py` | Auto-instalador |
| `analisis_nlp.md` | Explicación del impacto (limitado) de features NLP |
| `pyproject.toml` | Dependencias del proyecto |
| `apis.txt` | API keys (tracked solo por ser repo privado) |
| `data/` | ELO, partidos, predicciones, raw news/comments |
| `scripts/` | Scrapers y utilidades |

## Salida del modelo

Ambos modos muestran:
- Accuracy y log-loss en validación temporal
- Probabilidad local / empate / visitante para cada partido
- λ Poisson y top 5 scores exactos más probables
- Probabilidad de avance (empate repartido 50/50)
- Features NLP por equipo (solo con `--nlp`)

Guardan:
- `data/knockout_predictions.csv` — predicciones base
- `data/knockout_predictions_nlp.csv` — predicciones NLP

## Notas técnicas

- `MAX_DATE` se calcula como `hoy - 1 día` — los datos se actualizan solos al ejecutar
- Accuracy temporal: **58.81%**, log-loss: **0.8761** (Temporal split, 1634 partidos futuros)
- F1 macro: **0.4351**, MCC: **0.3246**, Top-2 accuracy: **83.11%**, ECE: **0.0106**
- El modelo casi no predice empates (0.3% predicho vs 23% real) — ver `evaluate_model.py`
- Creado con `uv init --python 3.12`
- En Manjaro, si Playwright falla: `sudo pacman -S atk at-spi2-atk cups libdrm libxkbcommon libxcomposite libxdamage libxrandr mesa nss pango cairo gtk3`
- Las features NLP existen para los 32 equipos pero tienen impacto limitado porque no hay datos NLP históricos para entrenar el meta-modelo (ver `analisis_nlp.md`)
- `apis.txt` se incluye en el repo porque es privado — si se hace público, descomentar en `.gitignore`
