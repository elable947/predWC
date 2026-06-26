# Análisis: ¿Por qué no cambian las predicciones al agregar features NLP?

## Conclusión rápida

Las predicciones del modelo stacking **no cambian** al añadir las 24 features NLP. Ambos archivos (`knockout_predictions.csv` y `knockout_predictions_nlp.csv`) son **idénticos** (mismo MD5).

## Explicación

### 1. No hay datos NLP históricos

El meta-modelo (Logistic Regression) se entrena con un split temporal: 80% train (partidos viejos) → 20% validation (partidos recientes). Las features NLP (news embeddings + YouTube sentiment) solo existen para los **32 equipos actuales del Mundial**, no para los ~8000 partidos históricos.

En el entrenamiento de validación, las features NLP se setean a **0.0** para todos los partidos:

```python
# stacking_model_nlp.py, línea 431
meta_val_nlp = np.zeros((len(y_val), len(NLP_FEATURE_COLS)))
```

Esto significa que el meta-modelo **aprendió pesos para las features NLP a partir de puros ceros**. Como todas las muestras de validación tienen NLP=0, el modelo no puede encontrar correlación entre NLP y resultado; los coeficientes de esas dimensiones son esencialmente ruido.

### 2. Features NLP ≈ ruido en predicción

Para los 16 partidos actuales, las features NLP sí tienen valores reales (ej: PC=-0.33, YT=+0.12). Pero como el meta-modelo fue entrenado con NLP=0 en todas partes, estos valores reales se multiplican por coeficientes que aprendieron de datos donde no había señal → el aporte es despreciable.

Los modelos base (RF, XGBoost, SVM) se entrenan solo con las 25 features deportivas, sin NLP. El meta-modelo recibe 9 probabilidades + 24 NLP. Las 9 probabilidades dominan porque fueron generadas por modelos que sí vieron patrones reales.

### 3. Lo que haría falta

Para que las features NLP tengan impacto real, necesitaríamos:

1. **Datos NLP históricos**: news articles y YouTube comments para cada partido en el dataset de entrenamiento (~8000 partidos desde 2018)
2. **Embeddings por partido**, no por equipo: la cobertura mediática de Brasil vs Argentina es distinta a Brasil vs Japón
3. **Pipeline re-entrenado**: el meta-modelo necesita ver correlaciones reales entre NLP y resultado

Sin eso, las features NLP son decorativas en el contexto actual del modelo.

### 4. Validación del modelo sin NLP

El modelo base obtiene:
- **Accuracy temporal**: 58.81%
- **Log-loss temporal**: 0.8761

Sobre 1634 partidos futuros (sin leakage), supera ampliamente el baseline aleatorio (33.3%). Las 25 features deportivas (ELO, forma reciente, head-to-head, peso del torneo) capturan la mayor parte de la señal predictiva.

## Conclusión

Las features NLP existen, se computan, se cargan, y el meta-modelo les asigna pesos — pero como no hay datos históricos con NLP, esos pesos se aprendieron de ceros y no afectan la salida. Es un requisito académico (variable NLP) implementado correctamente, pero que requiere recolección de datos históricos para ser verdaderamente efectivo.
