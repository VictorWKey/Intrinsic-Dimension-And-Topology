# Análisis de Dimensión Intrínseca usando Autoencoders

Este proyecto demuestra cómo usar **Autoencoders** para encontrar la **dimensión intrínseca** de datos de alta dimensionalidad, específicamente aplicado al dataset MNIST.

## 📁 Estructura del Proyecto

```
Intrinsic-Dimension/
├── src/
│   ├── models.py           # Arquitecturas de Autoencoder y VAE
│   ├── evaluation.py       # Métricas de evaluación
│   └── visualization.py    # Funciones de visualización
├── data/                   # Dataset MNIST (se descarga automáticamente)
├── results/                # Resultados y visualizaciones generadas
├── notebooks/
│   └── dimension_intrinseca_demo.ipynb  # Demo interactivo
├── main.py                 # Script principal de ejecución
└── requirements.txt        # Dependencias
```

## 🚀 Cómo Ejecutar

### Opción 1: Script Principal (Recomendado)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar análisis completo
python main.py
```

### Opción 2: Notebook Interactivo
```bash
# Instalar dependencias
pip install -r requirements.txt

# Abrir Jupyter
jupyter notebook notebooks/dimension_intrinseca_demo.ipynb
```

## ⚡ Qué hace el programa

1. **Carga MNIST**: Descarga y prepara el dataset de dígitos manuscritos (28×28 = 784 dimensiones)

2. **Entrena Autoencoders**: Crea modelos con diferentes dimensiones de cuello de botella (2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

3. **Evalúa Calidad**: Mide el error de reconstrucción para cada dimensión

4. **Encuentra Dimensión Óptima**: Usa múltiples métricas:
   - Método del codo
   - Umbral de estabilidad
   - Eficiencia de compresión
   - Rate-distortion

5. **Genera Visualizaciones**: Crea gráficas y tablas para presentación

## 📊 Resultados Esperados

- **Dimensión intrínseca estimada**: ~16-32 dimensiones
- **Factor de compresión**: ~25-50x
- **Visualizaciones**: Gráficas de error, reconstrucciones, espacio latente
- **Tablas de análisis**: Comparación de métricas

## 🎯 Para tu Presentación

El proyecto genera automáticamente:

- **Gráficas principales** en `results/`
- **Tablas de resultados** en formato CSV
- **Comparaciones visuales** de reconstrucciones
- **Análisis del espacio latente**
- **Conclusiones cuantitativas**

## 💡 Conceptos Demostrados

1. **Dimensión Intrínseca**: Cuántas dimensiones realmente necesitan los datos
2. **Autoencoders**: Cómo comprimen y reconstruyen información
3. **Método del Codo**: Técnica para encontrar el punto de inflexión óptimo
4. **Balance Compresión-Calidad**: Trade-off entre tamaño y fidelidad

## ⏱️ Tiempo de Ejecución

- **Script completo**: ~15-20 minutos
- **Notebook interactivo**: Variable (puedes ejecutar celdas individuales)

## 🔧 Personalización

Puedes modificar:
- `latent_dimensions` en `main.py` para probar otras dimensiones
- `num_epochs` para entrenar más/menos tiempo
- Dataset (cambiar MNIST por otro dataset)
- Arquitectura del autoencoder en `src/models.py`

¡Perfecto para demostrar conceptos de dimensión intrínseca en tu presentación! 🎓