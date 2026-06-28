# predWC — World Cup 2026 Knockout Predictor

Stacking ensemble (Random Forest + XGBoost + SVM → Logistic Regression) que predice los 16avos de final del Mundial 2026. Versión extendida con features NLP (news embeddings + YouTube sentiment).

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
1. Instala `uv` (gestor de paquetes Python) si no lo tienes
2. Crea entorno virtual e instala todas las dependencias con `uv sync`
3. Instala Playwright + Chromium (necesario para actualizar datos)

### Opción manual (paso a paso)

```bash
# 1. Instalar uv (si no lo tienes)
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

Una vez instalado, desde la carpeta del proyecto:

### Ejecutar el modelo base

```bash
uv run python stacking_model.py
```

Esto:
1. Descarga partidos históricos internacionales (~50k desde GitHub)
2. Construye 25 features por partido (ELO, forma reciente, h2h, peso del torneo)
3. Entrena 3 modelos base con split temporal (80/20 por fecha)
4. Entrena meta-modelo (Logistic Regression) con predicciones out-of-sample
5. Evalúa en validación temporal (~1630 partidos)
6. Predice los 16 cruces de 16avos y simula scores exactos (Poisson + Monte Carlo)

### Ejecutar el modelo con NLP

```bash
uv run python stacking_model_nlp.py
```

Incluye 24 features adicionales (11 componentes PCA de news embeddings + YouTube sentiment por equipo).

### Ver resultados gráficos

```bash
uv run python show_results.py       # resultados base
uv run python show_results.py --nlp # resultados NLP
```

Abre 4 ventanas: avance, probabilidades, confianza y scores Poisson.

## Actualizar datos

Los datasets ya vienen precargados, pero puedes actualizarlos:

### Actualizar todo (ELO + cruces)

```bash
uv run python scripts/update_data.py
```

### Actualizar noticias y NLP

```bash
# 1. Scrapea noticias (Wikipedia + BBC/ESPN)
uv run python scripts/fetch_team_news.py

# 2. Scrapea comentarios YouTube (necesita API key en apis.txt)
uv run python scripts/fetch_team_comments.py

# 3. Recalcula embeddings + PCA + sentiment
uv run python scripts/compute_team_embeddings.py
```

### Para re-entrenar con los datos actuales

```bash
uv run python stacking_model.py
uv run python stacking_model_nlp.py
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `stacking_model.py` | Modelo stacking base (25 features) |
| `stacking_model_nlp.py` | Modelo stacking + 24 features NLP |
| `show_results.py` | Visualización interactiva |
| `install_all.py` | Auto-instalador (recomendado) |
| `pyproject.toml` | Dependencias del proyecto |
| `analisis_nlp.md` | Por qué las features NLP no cambian las predicciones |
| `data/` | Rankings ELO, partidos, predicciones, raw news/comments |
| `scripts/` | Scrapers y utilidades para actualizar datos |

## Salida del modelo

Ambos modelos muestran:
- Accuracy y log-loss en validación temporal
- Probabilidad local / empate / visitante para cada partido
- λ Poisson y top 5 scores exactos más probables
- Probabilidad de avance (empate se reparte 50/50)

También guardan:
- `data/knockout_predictions.csv` — predicciones base
- `data/knockout_predictions_nlp.csv` — predicciones NLP

## Notas técnicas

- `MAX_DATE` se calcula como `hoy - 1 día` — los datos se actualizan solos al ejecutar
- Creado con `uv init --python 3.12` — el entorno usa exactamente esa versión
- En Manjaro, si Playwright falla, instalar: `sudo pacman -S atk at-spi2-atk cups libdrm libxkbcommon libxcomposite libxdamage libxrandr mesa nss pango cairo gtk3`
- Las features NLP no afectan las predicciones por falta de datos históricos con NLP (ver `analisis_nlp.md`)
