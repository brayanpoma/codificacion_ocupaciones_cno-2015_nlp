# Modelo de Concatenación: concat_bert-base-spanish-wwm-cased_20251121_063706

## Información del Modelo

- **Modelo Base**: dccuchile/bert-base-spanish-wwm-cased
- **Enfoque**: Concatenación de features como texto
- **Número de Clases**: 357
- **Fecha de Entrenamiento**: 2025-11-21 08:11:13

## Formato de Entrada

Este modelo usa CONCATENACIÓN de features:

```
Input Original:
  texto: "vendedor de abarrotes"
  edad: 35
  nivel: 5
  desempeño: 1

Texto Concatenado:
  "vendedor de abarrotes , edad: 35 años , educación: secundaria completa , desempeño: independiente"
```

## Configuración

- **Include labels**: True
- **Separator**: " , "
- **Max length**: 128
- **Batch size**: 16
- **Learning rate**: 2e-05
- **Epochs**: 3

## Resultados (Test Set)

- **Accuracy**: 0.9456
- **F1 Weighted**: 0.9425
- **F1 Macro**: 0.6239

## Uso del Modelo

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Cargar modelo y tokenizer
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/classification_coding_open_ended_occupational_responses_ENAHO/results/bertin_concatenado/concat_bert-base-spanish-wwm-cased_20251121_063706/final_model")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/classification_coding_open_ended_occupational_responses_ENAHO/results/bertin_concatenado/concat_bert-base-spanish-wwm-cased_20251121_063706/final_model")

# Cargar artefactos
with open("/content/drive/MyDrive/classification_coding_open_ended_occupational_responses_ENAHO/results/bertin_concatenado/concat_bert-base-spanish-wwm-cased_20251121_063706/artifacts.pkl", 'rb') as f:
    artifacts = pickle.load(f)

# Preparar texto
def concatenate_features(texto, edad, nivel, desempeno):
    nivel_desc = artifacts['nivel_educativo_map'][nivel]
    desemp_desc = artifacts['desempeno_map'][desempeno]

    return f"{texto} , edad: {edad} años , educación: {nivel_desc} , desempeño: {desemp_desc}"

# Ejemplo
texto_concat = concatenate_features(
    texto="vendedor de abarrotes",
    edad=35,
    nivel=5,
    desempeno=1
)

# Tokenizar y predecir
inputs = tokenizer(texto_concat, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
predicted_label = artifacts['id2label'][str(predicted_class)]

print(f"Predicción: {predicted_label}")
```

## Archivos

- `pytorch_model.bin`: Pesos del modelo
- `config.json`: Configuración del modelo
- `tokenizer.json`: Tokenizer
- `artifacts.pkl`: Mapeos y metadata
- `test_metrics.json`: Métricas de evaluación
- `classification_report.txt`: Reporte detallado
- `error_analysis.csv`: Análisis de errores

## Comparación con Multimodal

Este modelo usa concatenación simple (todas las features como texto).
Para mejor performance, considera el enfoque multimodal que preserva la información numérica.
