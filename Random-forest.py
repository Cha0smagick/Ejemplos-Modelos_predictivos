# -*- coding: utf-8 -*-
"""
Script Didáctico de Clasificación con Bosque Aleatorio (Random Forest).

Objetivo:
- Implementar un modelo de Machine Learning para predecir si un cliente realizará una compra (clasificación binaria).
- Explicar cada paso del proceso de una manera sencilla y clara para principiantes,
  utilizando un conjunto de datos sintético.

Analogía:
Un Bosque Aleatorio es como un jurado de expertos. Cada "experto" es un "Árbol de Decisión" que
toma una decisión basándose en una parte de la información. La decisión final del bosque
es la opción más votada por todos los árboles, lo que la hace más robusta y confiable.

Librerías utilizadas:
- pandas: Para la manipulación de datos en tablas (DataFrames).
- numpy: Para operaciones numéricas y la creación de datos sintéticos.
- scikit-learn: Para implementar el modelo, preprocesar datos y evaluar el rendimiento.
"""

# =============================================================================
# FASE 0: IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# =============================================================================
# Antes de empezar, importamos las herramientas que vamos a necesitar.
# Es como reunir los ingredientes antes de empezar a cocinar.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# =============================================================================
# FASE 1: PREPARACIÓN Y SIMULACIÓN DE DATOS
# =============================================================================
# En un proyecto real, aquí cargaríamos nuestros datos desde un archivo.
# Como no tenemos un archivo, vamos a generar datos de ejemplo (sintéticos)
# para simular un escenario real.

print("--- FASE 1: Preparación y Simulación de Datos ---")

# --- Bloque de Carga de Datos (Simulado) ---
# En un caso real, reemplazarías este bloque por:
# try:
#     df = pd.read_csv('Datos_de_prueba_expandidos.csv')
#     print("Archivo 'Datos_de_prueba_expandidos.csv' cargado exitosamente.")
# except FileNotFoundError:
#     print("Error: El archivo no se encontró. Se usarán datos sintéticos.")
#     # Aquí podría ir la generación de datos como fallback o simplemente terminar el script.

# Generación de datos sintéticos para el ejemplo
np.random.seed(42) # Usamos una semilla para que los resultados sean siempre los mismos.
num_filas = 200
paises = ['Mexico', 'USA', 'Colombia', 'España']
generos = ['Masculino', 'Femenino']

datos = {
    'edad': np.random.randint(18, 70, size=num_filas),
    'ingreso_anual_miles': np.random.randint(20, 150, size=num_filas),
    'puntuacion_credito': np.random.randint(300, 850, size=num_filas),
    'pais': np.random.choice(paises, size=num_filas),
    'genero': np.random.choice(generos, size=num_filas)
}
df = pd.DataFrame(datos)

# Creación de la variable objetivo 'compra_realizada' (0 = No, 1 = Sí)
# Hacemos que la probabilidad de compra dependa de las otras variables.
probabilidad_compra = (df['puntuacion_credito'] / 850) * (df['ingreso_anual_miles'] / 150)
df['compra_realizada'] = (probabilidad_compra > np.random.rand(num_filas) * 0.4).astype(int)

print("Se han generado 200 filas de datos sintéticos para el ejemplo.")
print("\nPrimeras 5 filas del dataset generado:")
print(df.head())

# --- Separación de Características (X) y Variable Objetivo (y) ---
# 'X' son las columnas que usaremos para predecir.
# 'y' es la columna que queremos predecir ('compra_realizada').

X = df.drop('compra_realizada', axis=1) # Quitamos la columna objetivo para obtener las características.
y = df['compra_realizada']             # Seleccionamos solo la columna objetivo.

print("\nCaracterísticas (X) y variable objetivo (y) separadas.")


# =============================================================================
# FASE 2: PREPROCESAMIENTO BÁSICO
# =============================================================================
# Los modelos de Machine Learning necesitan que los datos estén en formato numérico.
# Convertiremos las columnas de texto (categóricas) a números.

print("\n--- FASE 2: Preprocesamiento Básico ---")

# --- Paso 2.1: Identificar columnas categóricas y numéricas ---
variables_categoricas = X.select_dtypes(include=['object']).columns
variables_numericas = X.select_dtypes(include=['int64', 'float64']).columns
print(f"Variables categóricas identificadas: {list(variables_categoricas)}")
print(f"Variables numéricas identificadas: {list(variables_numericas)}")

# --- Paso 2.2: Crear el transformador para las columnas ---
# Usaremos OneHotEncoder para convertir las categorías en columnas de 0s y 1s.
# Por ejemplo, 'pais_Mexico', 'pais_USA', etc.
# 'remainder='passthrough'' asegura que las columnas numéricas no se eliminen.
preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), variables_categoricas)
    ],
    remainder='passthrough' # Mantiene las columnas no especificadas (las numéricas)
)

# --- Paso 2.3: Dividir los datos en conjuntos de Entrenamiento y Prueba ---
# ¡Este es un paso CRÍTICO!
# Entrenamos el modelo con el 70% de los datos (entrenamiento) y lo evaluamos con
# el 30% restante, que el modelo nunca ha visto (prueba).
# Esto nos dice si el modelo realmente "aprendió" o si solo "memorizó".
# `random_state=42` asegura que la división sea siempre la misma para poder reproducir los resultados.
# `stratify=y` mantiene la misma proporción de 'compradores' y 'no compradores' en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDatos divididos: {len(X_train)} filas para entrenamiento y {len(X_test)} para prueba.")


# =============================================================================
# FASE 3: INSTANCIACIÓN Y ENTRENAMIENTO DEL MODELO
# =============================================================================
# Ahora creamos nuestro "jurado de expertos" y lo entrenamos con los datos de entrenamiento.

print("\n--- FASE 3: Instanciación y Entrenamiento del Modelo ---")

# --- Paso 3.1: Aplicar el preprocesamiento ---
# Ajustamos el preprocesador con los datos de entrenamiento y los transformamos.
X_train_procesado = preprocesador.fit_transform(X_train)
# Usamos el preprocesador ya ajustado para transformar los datos de prueba.
X_test_procesado = preprocesador.transform(X_test)

print("Variables categóricas transformadas a formato numérico.")

# --- Paso 3.2: Crear y entrenar el modelo de Bosque Aleatorio ---
# `n_estimators=100` significa que nuestro bosque tendrá 100 árboles (expertos).
# `random_state=42` para que la aleatoriedad interna del modelo sea reproducible.
# `class_weight='balanced'` ayuda al modelo a prestar la misma atención a ambas clases (compradores y no compradores).
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Entrenamos el modelo con los datos de entrenamiento ya procesados.
# El modelo aprenderá los patrones que relacionan las características con la decisión de compra.
modelo_rf.fit(X_train_procesado, y_train)

print("Modelo de Bosque Aleatorio entrenado exitosamente.")


# =============================================================================
# FASE 4: PREDICCIÓN
# =============================================================================
# Usamos nuestro modelo ya entrenado para que haga predicciones sobre el conjunto de prueba.

print("\n--- FASE 4: Realizando Predicciones ---")

# El modelo usará lo que aprendió para predecir si los clientes del conjunto de prueba comprarán o no.
y_pred = modelo_rf.predict(X_test_procesado)

print("Predicciones realizadas sobre el conjunto de prueba.")


# =============================================================================
# FASE 5: EVALUACIÓN DEL MODELO
# =============================================================================
# ¿Qué tan bueno es nuestro modelo? Medimos su rendimiento con los datos de prueba.

print("\n--- FASE 5: Evaluación del Modelo ---")

# --- Exactitud (Accuracy) ---
# ¿Qué porcentaje de predicciones fueron correctas?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud (Accuracy): {accuracy:.2f} ({accuracy:.2%})")
print("Esto significa que el modelo acertó en el porcentaje de predicciones mostrado arriba.")

# --- Matriz de Confusión ---
# Nos da un desglose más detallado de los aciertos y errores.
# [[Verdaderos Negativos, Falsos Positivos],
#  [Falsos Negativos,   Verdaderos Positivos]]
print("\nMatriz de Confusión:")
# y_test son las etiquetas verdaderas, y_pred son las que el modelo predijo.
matriz = confusion_matrix(y_test, y_pred)
print(matriz)
print(f" - Verdaderos Negativos (TN): {matriz[0][0]} (predijo 'No Compra' y acertó)")
print(f" - Falsos Positivos (FP):    {matriz[0][1]} (predijo 'Compra' y se equivocó)")
print(f" - Falsos Negativos (FN):    {matriz[1][0]} (predijo 'No Compra' y se equivocó)")
print(f" - Verdaderos Positivos (TP): {matriz[1][1]} (predijo 'Compra' y acertó)")

# --- Reporte de Clasificación ---
# Muestra métricas clave como Precisión, Recall y F1-Score para cada clase.
# - Precisión: De los que predijo como 'Compra', ¿cuántos acertó?
# - Recall (Sensibilidad): De todos los que realmente eran 'Compra', ¿cuántos encontró?
# - F1-Score: Es una media armónica de Precisión y Recall.
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Compra (0)', 'Compra (1)']))

