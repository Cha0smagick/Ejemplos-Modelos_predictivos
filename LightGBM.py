# -*- coding: utf-8 -*-
"""
Script Didáctico de LightGBM para Clasificación.

Objetivo:
- Implementar un modelo predictivo usando la librería LightGBM para clasificar
  si un cliente comprará un producto o no.
- Explicar cada paso de forma extremadamente detallada y sencilla, ideal para
  principiantes sin experiencia previa en programación, estadística o machine learning.

Analogía de LightGBM:
Imagina que LightGBM es un detective increíblemente rápido y eficiente.
A diferencia de otros detectives que revisan cada pista una por una (como otros modelos),
LightGBM se enfoca primero en las pistas más "prometedoras", las que parecen
llevar más rápido a la solución. Es como si, al investigar un caso, en lugar de
interrogar a todos los testigos en orden, fuera directamente a hablar con el que
parece saber más. Esta eficiencia lo hace muy rápido, especialmente con muchos datos.
Al igual que XGBoost, también aprende de sus errores, pero lo hace de una manera
más "vertical" (centrándose en las hojas del árbol con más error), lo que acelera el proceso.

Librerías utilizadas:
- os: Para interactuar con el sistema operativo, como verificar si un archivo existe.
- pandas: Para cargar y manipular los datos en tablas (DataFrames).
- numpy: Para operaciones numéricas y la creación de datos de ejemplo.
- scikit-learn: Para dividir los datos, preprocesarlos y evaluar el modelo.
- lightgbm: La librería que contiene el algoritmo de LightGBM.
"""

# =============================================================================
# FASE 0: PREPARACIÓN E IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# Antes de empezar, reunimos todas las herramientas (librerías) que vamos a necesitar.
# Es como preparar los ingredientes y utensilios antes de cocinar una receta.

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Bloque de Simulación de Datos ---
# Verificamos si el archivo de datos ya existe. Si no, creamos uno de ejemplo.
# Esto asegura que el script siempre pueda ejecutarse, incluso si no tienes el archivo.
file_name = 'Datos_de_prueba_expandidos.csv'
if not os.path.exists(file_name):
    print(f"El archivo '{file_name}' no se encontró. Creando datos de ejemplo...")
    # Usamos una "semilla" para que los datos aleatorios sean siempre los mismos.
    np.random.seed(42)
    num_filas = 500
    paises = ['Mexico', 'Colombia', 'España', 'Argentina', 'Chile']
    
    datos = {
        'Edad': np.random.randint(18, 65, size=num_filas),
        'Pais': np.random.choice(paises, size=num_filas),
        'Ingresos_Anuales': np.random.uniform(15000, 90000, size=num_filas).round(2),
        'Tiempo_en_Sitio_Min': np.random.uniform(5, 60, size=num_filas).round(2),
        'Dispositivo': np.random.choice(['Movil', 'Desktop'], size=num_filas, p=[0.7, 0.3]),
        'Resultado': np.random.choice(['Aceptada', 'Rechazada'], size=num_filas, p=[0.6, 0.4])
    }
    df_simulado = pd.DataFrame(datos)
    
    # Introducimos algunos valores faltantes (NaN) a propósito para mostrar cómo manejarlos.
    for col in ['Ingresos_Anuales', 'Tiempo_en_Sitio_Min']:
        idx = df_simulado.sample(frac=0.05).index
        df_simulado.loc[idx, col] = np.nan
        
    df_simulado.to_csv(file_name, index=False, encoding='utf-8')
    print(f"Archivo '{file_name}' creado exitosamente con {num_filas} filas.")


# =============================================================================
# FASE 1: CARGA Y EXPLORACIÓN DE DATOS (EDA BÁSICO)
# =============================================================================
# El primer paso es cargar nuestros datos y echarles un vistazo para entenderlos.
# Es como leer la lista de ingredientes y ver qué tenemos antes de empezar a cocinar.

print("\n--- FASE 1: Carga y Exploración de Datos ---")

# Cargamos el archivo CSV en un DataFrame de pandas.
# Un DataFrame es, en esencia, una tabla como las que verías en Excel.
try:
    df = pd.read_csv(file_name, encoding='utf-8')
    print(f"Archivo '{file_name}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{file_name}' no se encontró. Asegúrate de que esté en la misma carpeta.")
    exit() # Si no hay datos, no podemos continuar.

# Mostramos las primeras 5 filas para tener una idea de cómo se ven los datos.
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# Mostramos información general: número de filas, columnas y tipo de dato de cada una.
# Esto nos ayuda a identificar qué columnas son numéricas y cuáles son texto (categóricas).
print("\nInformación general del dataset:")
df.info()


# =============================================================================
# FASE 2: PREPROCESAMIENTO DE DATOS
# =============================================================================
# Los modelos de Machine Learning son como calculadoras muy avanzadas: solo entienden números.
# No pueden trabajar con texto ('Mexico', 'Movil') o valores vacíos (NaN).
# En esta fase, "limpiamos y traducimos" nuestros datos a un lenguaje que el modelo pueda entender.

print("\n--- FASE 2: Preprocesamiento de Datos ---")

# --- Paso 2.1: Identificar y separar la variable objetivo (y) y las predictoras (X) ---
# 'y' es lo que queremos predecir ('Resultado').
# 'X' son todas las demás columnas que usaremos como pistas para hacer la predicción.
X = df.drop('Resultado', axis=1)
y = df['Resultado']

print("\nDatos separados en características (X) y variable objetivo (y).")

# --- Paso 2.2: Manejo de Valores Faltantes (Imputación) ---
# Nuestro 'df.info()' nos mostró que hay valores nulos (faltantes).
# Una estrategia simple es reemplazarlos con la media (el promedio) de su columna.
# Analogía: Si no sabes la edad de una persona en un grupo, una suposición razonable es la edad promedio del grupo.
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        media = X[col].mean()
        X[col].fillna(media, inplace=True)
        print(f"Valores faltantes en '{col}' rellenados con la media ({media:.2f}).")

# --- Paso 2.3: Codificación de Variables Categóricas ---
# Convertimos las columnas de texto a números. LightGBM puede manejar esto internamente,
# pero hacerlo explícitamente nos da más control y es una buena práctica.
# Usaremos Label Encoding: asigna un número único a cada categoría (ej: 'Movil'=0, 'Desktop'=1).
# Es como crear un código numérico para cada palabra.
variables_categoricas = X.select_dtypes(include=['object']).columns
for col in variables_categoricas:
    # Creamos un codificador para cada columna
    le = LabelEncoder()
    # Lo ajustamos a los datos y los transformamos a números
    X[col] = le.fit_transform(X[col])
    print(f"Variable categórica '{col}' codificada a formato numérico.")
    # También es importante decirle a LightGBM que estas columnas son categóricas
    X[col] = X[col].astype('category')

# También codificamos la variable objetivo 'y' (ej: 'Rechazada'=0, 'Aceptada'=1)
le_y = LabelEncoder()
y = le_y.fit_transform(y)
print("Variable objetivo 'Resultado' codificada a 0 y 1.")

print("\nVista de los datos preprocesados (ahora todo es numérico):")
print(X.head())

# --- Paso 2.4: Dividir los datos en Entrenamiento y Prueba ---
# ¡Este es un paso CRÍTICO! Es como prepararse para un examen.
# - Set de Entrenamiento (70%): El material de estudio que le damos al modelo para que aprenda.
# - Set de Prueba (30%): Un examen sorpresa con preguntas que el modelo nunca ha visto.
# Esto nos permite evaluar si el modelo realmente "aprendió" a generalizar o si solo
# "memorizó" las respuestas (lo que se llama "sobreajuste" u "overfitting").
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# `test_size=0.3` significa que el 30% de los datos se usarán para la prueba.
# `random_state=42` es como elegir siempre el mismo número en un sorteo, para que la división sea reproducible.
# `stratify=y` asegura que la proporción de 'Aceptadas' y 'Rechazadas' sea la misma en ambos sets.

print(f"\nDatos divididos: {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")


# =============================================================================
# FASE 3: CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO LIGHTGBM
# =============================================================================
# Ahora que los datos están listos, creamos y entrenamos a nuestro "detective" LightGBM.

print("\n--- FASE 3: Entrenamiento del Modelo LightGBM ---")

# Creamos el modelo de clasificación de LightGBM.
# Le pasamos algunos parámetros para decirle cómo debe aprender.
modelo_lgbm = lgb.LGBMClassifier(
    objective='binary',      # 'binary' porque es un problema de dos clases (Aceptada/Rechazada).
    metric='binary_logloss', # La métrica que el modelo intentará optimizar durante el entrenamiento.
    random_state=42          # Para que los resultados sean reproducibles.
)

# ¡El entrenamiento! Aquí es donde el modelo "estudia" los datos de entrenamiento.
# Aprende los patrones que conectan las características (X_train) con el resultado (y_train).
# `categorical_feature='auto'` le permite a LightGBM usar eficientemente las columnas que marcamos como 'category'.
modelo_lgbm.fit(X_train, y_train, categorical_feature='auto')

print("Modelo LightGBM entrenado exitosamente.")


# =============================================================================
# FASE 4: EVALUACIÓN Y EXPLICACIÓN DE RESULTADOS
# =============================================================================
# ¿Qué tan bueno es nuestro modelo? Vamos a medir su rendimiento en el "examen sorpresa".

print("\n--- FASE 4: Evaluación del Modelo ---")

# Usamos el modelo entrenado para hacer predicciones sobre los datos de prueba.
y_pred = modelo_lgbm.predict(X_test)

# --- Métricas de Rendimiento ---
# Medimos qué tan bien lo hizo el modelo.

# Accuracy (Exactitud): ¿Qué porcentaje de predicciones fueron correctas?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Exactitud) en el conjunto de prueba: {accuracy:.4f}")
print(f"Esto significa que el modelo acertó en el {accuracy:.2%} de las predicciones.")

# F1-Score: Una métrica que combina Precisión y Recall. Es muy útil cuando las clases
# están desbalanceadas (hay muchos más de un tipo que de otro).
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")
print("Un F1-Score cercano a 1 indica un buen equilibrio entre no cometer errores y encontrar todos los casos positivos.")

# Reporte de Clasificación: Un análisis más profundo.
# - Precisión: De todas las veces que el modelo predijo 'Aceptada', ¿cuántas veces acertó?
# - Recall (Exhaustividad): De todas las transacciones que realmente eran 'Aceptada', ¿cuántas encontró el modelo?
print("\nReporte de Clasificación detallado:")
# Usamos los nombres originales de las clases para que el reporte sea más fácil de leer.
target_names = le_y.classes_
print(classification_report(y_test, y_pred, target_names=target_names))


# --- Importancia de las Características (Feature Importance) ---
# ¿En qué pistas se fijó más nuestro "detective" para tomar sus decisiones?
# Esto nos ayuda a entender qué variables son más importantes para el modelo.
print("\n--- Importancia de las Características ---")
print("Mostrando qué variables fueron más influyentes para las predicciones del modelo:")

# LightGBM nos puede decir qué características usó más.
lgb.plot_importance(modelo_lgbm, figsize=(10, 6), title="Importancia de las Características (LightGBM)")
plt.show()

# También podemos mostrarlo en una tabla para mayor claridad.
df_importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo_lgbm.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nTabla de importancia de las variables:")
print(df_importancia)

print("\n" + "="*60)
print("FIN DEL SCRIPT")
print("="*60)

"""
Conclusión:

¡Felicidades! Has implementado y evaluado un modelo de clasificación con LightGBM.

1. Preparación: Aprendiste a simular datos, cargarlos y realizar una exploración inicial.

2. Preprocesamiento: Viste cómo manejar valores faltantes y convertir texto a números
   para que el modelo pueda procesarlos.

3. Entrenamiento: Entrenaste un modelo LightGBM, una herramienta potente y rápida.

4. Evaluación: Mediste el rendimiento del modelo con métricas como Accuracy y F1-Score,
   y aprendiste a interpretar un reporte de clasificación.

5. Interpretación: Descubriste cómo el modelo "piensa" al analizar la importancia de
   las características, lo que te da una idea de qué factores son más decisivos en el problema.

Este script es un excelente punto de partida. El siguiente paso en tu viaje por el
Machine Learning podría ser aprender a optimizar los parámetros del modelo (ajuste de
hiperparámetros) para hacerlo aún más preciso.
"""
