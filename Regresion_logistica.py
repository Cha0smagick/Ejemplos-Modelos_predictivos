# -*- coding: utf-8 -*-
"""
Script Didáctico de Regresión Logística para Clasificación Binaria.

Objetivo:
- Implementar un modelo de Machine Learning para predecir si una transacción será 'Aceptada' o 'Rechazada'.
- Explicar cada paso del proceso, desde la carga de datos hasta la evaluación del modelo,
  de una manera sencilla y clara para principiantes.

Librerías utilizadas:
- pandas: Para la manipulación y análisis de datos (leer el CSV, transformar columnas, etc.).
- scikit-learn: Para implementar el modelo de Regresión Logística y las herramientas de preprocesamiento.
"""

# =============================================================================
# FASE 0: IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# =============================================================================
# Antes de empezar, importamos las herramientas que vamos a necesitar.
# Es como reunir los ingredientes antes de empezar a cocinar.

import pandas as pd  # Se usa para trabajar con tablas de datos (DataFrames).
from sklearn.model_selection import train_test_split  # Para dividir nuestros datos en entrenamiento y prueba.
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Para normalizar datos numéricos y convertir categóricos.
from sklearn.compose import ColumnTransformer # Para aplicar diferentes transformaciones a diferentes columnas.
from sklearn.linear_model import LogisticRegression  # El modelo de clasificación que vamos a entrenar.
from sklearn.metrics import accuracy_score, classification_report  # Para evaluar qué tan bueno es nuestro modelo.
from sklearn.pipeline import Pipeline # Para organizar nuestro flujo de preprocesamiento y modelado.


# =============================================================================
# FASE 1: CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================
# El primer paso en cualquier proyecto de Machine Learning es entender los datos con los que trabajamos.

print("--- FASE 1: Carga y Exploración de Datos ---")

# Cargamos el archivo CSV en un DataFrame de pandas.
# Un DataFrame es, en esencia, una tabla como las que verías en Excel.
try:
    df = pd.read_csv('Datos_de_prueba_expandidos.csv', encoding='utf-8')
    print("Archivo 'Datos_de_prueba_expandidos.csv' cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'Datos_de_prueba_expandidos.csv' no se encontró.")
    print("Asegúrate de que el script y el archivo CSV estén en el mismo directorio.")
    exit() # Si no hay datos, no podemos continuar.

# Mostramos las primeras 5 filas para tener una idea de cómo se ven los datos.
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# Mostramos información general del DataFrame: número de filas, columnas y tipo de dato de cada columna.
print("\nInformación general del dataset:")
df.info()


# =============================================================================
# FASE 2: PREPROCESAMIENTO DE DATOS
# =============================================================================
# Los modelos de Machine Learning no pueden trabajar con datos "crudos" (texto, fechas, etc.).
# Necesitamos limpiar y transformar los datos para que el modelo los pueda entender.
# Esta es una de las fases más importantes y que más tiempo consume.

print("\n--- FASE 2: Preprocesamiento de Datos ---")

# --- Paso 2.1: Definir la variable objetivo (y) y las predictoras (X) ---

# La columna 'Resultado' es lo que queremos predecir. La convertiremos a un formato numérico.
# 'Aceptada' será 1 (éxito) y 'Rechazada' será 0 (fracaso).
# Esto se llama "mapeo" o "codificación de etiquetas".
df['Resultado'] = df['Resultado'].map({'Aceptada': 1, 'Rechazada': 0})

# Verificamos que no haya valores nulos en la columna objetivo después del mapeo.
if df['Resultado'].isnull().any():
    print("\nSe encontraron valores no esperados en la columna 'Resultado'. Abortando.")
    # Esto podría pasar si hubiera un valor diferente a 'Aceptada' o 'Rechazada'.
    exit()

print("\nVariable objetivo 'Resultado' ha sido transformada a formato binario (1 y 0).")

# --- Paso 2.2: Eliminar columnas no relevantes ---
# Algunas columnas no ayudan a predecir el resultado, como los identificadores únicos.
# Incluirlas podría confundir al modelo.
columnas_a_eliminar = ['Número de tarjeta de crédito', 'Nombre de la persona', 'Correo electrónico']
df = df.drop(columns=columnas_a_eliminar)
print(f"Columnas irrelevantes eliminadas: {columnas_a_eliminar}")

# Separamos el DataFrame en 'X' (las características o variables predictoras) y 'y' (la variable objetivo).
X = df.drop('Resultado', axis=1)
y = df['Resultado']

# --- Paso 2.3: Identificar tipos de variables (Numéricas y Categóricas) ---
# El modelo trata diferente a los números (ej. Monto) que a las categorías (ej. País).

# Seleccionamos las columnas que tienen datos de tipo numérico.
variables_numericas = X.select_dtypes(include=['int64', 'float64']).columns
print(f"\nVariables numéricas identificadas: {list(variables_numericas)}")

# Seleccionamos las columnas que tienen datos de tipo texto (categóricas).
variables_categoricas = X.select_dtypes(include=['object']).columns
print(f"Variables categóricas identificadas: {list(variables_categoricas)}")

# --- Paso 2.4: Dividir los datos en conjuntos de Entrenamiento y Prueba ---
# ¡Este es un paso CRÍTICO!
# Entrenamos el modelo con una parte de los datos (entrenamiento) y lo evaluamos con
# datos que nunca ha visto (prueba). Esto nos dice si el modelo realmente "aprendió"
# o si solo "memorizó" las respuestas del set de entrenamiento (lo que se conoce como sobreajuste).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# test_size=0.3 significa que el 30% de los datos se usarán para la prueba.
# random_state=42 asegura que la división sea siempre la misma, para que los resultados sean reproducibles.
# stratify=y asegura que la proporción de 1s y 0s sea la misma en el set de entrenamiento y prueba.

print(f"\nDatos divididos: {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")

# --- Paso 2.5: Crear un Pipeline de Preprocesamiento ---
# Un Pipeline nos permite encadenar los pasos de transformación de datos.
# 1. Para las variables numéricas: las escalaremos.
# 2. Para las variables categóricas: aplicaremos One-Hot Encoding.

# Escalado de Características (para variables numéricas):
# Pone todas las variables numéricas en una escala similar (media 0, desviación estándar 1).
# Esto ayuda a que el modelo converja más rápido y no le dé más importancia a una variable solo porque sus números son más grandes.
transformador_numerico = StandardScaler()

# One-Hot Encoding (para variables categóricas):
# Convierte categorías como 'USA' o 'Canada' en columnas de 0s y 1s.
# Por ejemplo, la columna 'País' se convertirá en 'País_USA', 'País_Canada', etc.
# handle_unknown='ignore' evita errores si en el futuro aparecen categorías no vistas en el entrenamiento.
transformador_categorico = OneHotEncoder(handle_unknown='ignore')

# Usamos ColumnTransformer para aplicar el transformador correcto a cada tipo de columna.
preprocesador = ColumnTransformer(
    transformers=[
        ('num', transformador_numerico, variables_numericas),
        ('cat', transformador_categorico, variables_categoricas)
    ])


# =============================================================================
# FASE 3: ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA
# =============================================================================
# Ahora que los datos están listos, podemos entrenar nuestro modelo.

print("\n--- FASE 3: Entrenamiento del Modelo ---")

# Creamos el modelo de Regresión Logística.
# `random_state=42` para reproducibilidad.
# `max_iter=1000` para asegurar que el modelo tenga suficientes iteraciones para converger.
modelo_logistico = LogisticRegression(random_state=42, max_iter=1000)

# Creamos el pipeline final que une el preprocesamiento y el modelo.
# Cuando llamemos a `fit` en este pipeline, hará lo siguiente:
# 1. Aplicará el `preprocesador` a los datos de entrenamiento (X_train).
#    - Escalará las variables numéricas.
#    - Codificará las variables categóricas.
# 2. Entrenará el `modelo_logistico` con los datos ya preprocesados.
pipeline_completo = Pipeline(steps=[('preprocesador', preprocesador),
                                  ('clasificador', modelo_logistico)])

# ¡Entrenamos el pipeline completo con los datos de entrenamiento!
pipeline_completo.fit(X_train, y_train)

print("Modelo de Regresión Logística entrenado exitosamente.")


# =============================================================================
# FASE 4: EVALUACIÓN DEL MODELO
# =============================================================================
# ¿Qué tan bueno es nuestro modelo? Vamos a medir su rendimiento en el conjunto de prueba.

print("\n--- FASE 4: Evaluación del Modelo ---")

# Hacemos predicciones sobre el conjunto de prueba (datos que el modelo no ha visto).
# El pipeline se encarga automáticamente de aplicar las mismas transformaciones (escalado, one-hot)
# que se aprendieron del conjunto de entrenamiento.
y_pred = pipeline_completo.predict(X_test)

# Calculamos la exactitud (Accuracy): ¿Qué porcentaje de predicciones fueron correctas?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Exactitud) en el conjunto de prueba: {accuracy:.2f}")
print(f"Esto significa que el modelo acertó en el {accuracy:.2%} de las predicciones.")

# Generamos un reporte de clasificación más detallado.
# - Precisión: De todas las veces que el modelo predijo '1', ¿cuántas veces acertó?
# - Exhaustividad (Recall): De todos los verdaderos '1' que había, ¿cuántos encontró el modelo?
# - F1-Score: Una media armónica de precisión y exhaustividad. Es útil para clases desbalanceadas.
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Rechazada (0)', 'Aceptada (1)']))


# =============================================================================
# FASE 5: INTERPRETACIÓN DE LOS COEFICIENTES
# =============================================================================
# ¿Qué variables son más importantes para el modelo?

print("\n--- FASE 5: Interpretación de Coeficientes ---")

# Extraemos los nombres de las características después del OneHotEncoding
nombres_categoricas_transformadas = list(pipeline_completo.named_steps['preprocesador'].named_transformers_['cat'].get_feature_names_out(variables_categoricas))

# Combinamos los nombres de las variables numéricas y las nuevas categóricas
todos_los_nombres = list(variables_numericas) + nombres_categoricas_transformadas

# Obtenemos los coeficientes (pesos) que el modelo aprendió para cada variable.
coeficientes = pipeline_completo.named_steps['clasificador'].coef_[0]

# Creamos un DataFrame para ver los coeficientes de forma ordenada.
df_coeficientes = pd.DataFrame({'Variable': todos_los_nombres, 'Coeficiente': coeficientes})
df_coeficientes['Impacto'] = df_coeficientes['Coeficiente'].apply(lambda x: 'Positivo' if x > 0 else 'Negativo')

print("Interpretación de los coeficientes del modelo:")
print("Un coeficiente POSITIVO para una variable significa que al aumentar el valor de esa variable, AUMENTA la probabilidad de que la transacción sea 'Aceptada (1)'.")
print("Un coeficiente NEGATIVO para una variable significa que al aumentar el valor de esa variable, DISMINUYE la probabilidad de que la transacción sea 'Aceptada (1)'.")

# Mostramos los 5 coeficientes con mayor impacto positivo y negativo.
df_coeficientes_sorted = df_coeficientes.sort_values('Coeficiente', ascending=False)

print("\nTop 5 variables con mayor impacto POSITIVO (más probable que sea Aceptada):")
print(df_coeficientes_sorted.head(5))

print("\nTop 5 variables con mayor impacto NEGATIVO (más probable que sea Rechazada):")
print(df_coeficientes_sorted.tail(5))
