# -*- coding: utf-8 -*-
"""
Script Didáctico de XGBoost para Clasificación y Regresión.

Objetivo:
- Implementar dos modelos predictivos usando la librería XGBoost:
  1. Un clasificador para predecir si una transacción será 'Aceptada' o 'Rechazada'.
  2. Un regresor para predecir el 'Monto de la transacción'.
- Explicar cada paso de forma extremadamente detallada y sencilla, ideal para principiantes
  sin experiencia previa en programación, estadística o machine learning.

Analogía de XGBoost:
Imagina que estás formando un equipo de "expertos" para tomar una decisión.
- El primer experto hace una predicción inicial. Es probable que cometa algunos errores.
- El segundo experto se enfoca específicamente en los errores que cometió el primero y trata de corregirlos.
- El tercer experto se enfoca en los errores restantes que dejaron los dos primeros.
- Este proceso continúa, y cada nuevo experto que se une al equipo es un especialista en corregir
  los errores de los anteriores.
XGBoost (Extreme Gradient Boosting) funciona de esta manera: construye modelos (árboles de decisión)
de forma secuencial, donde cada nuevo árbol aprende de los errores del anterior. El resultado final
es un equipo de expertos altamente preciso y robusto.

Librerías utilizadas:
- pandas: Para cargar y manipular los datos en tablas (DataFrames).
- scikit-learn: Para dividir los datos y evaluar los modelos.
- xgboost: La librería que contiene los algoritmos de XGBoost.
"""

# =============================================================================
# FASE 0: IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# =============================================================================
# Antes de empezar, importamos las herramientas que vamos a necesitar.
# Es como reunir los ingredientes y utensilios antes de empezar a cocinar.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import numpy as np # Se usa para la raíz cuadrada en RMSE.


# =============================================================================
# FASE 1: CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================
# El primer paso en cualquier proyecto de Machine Learning es cargar y entender
# los datos con los que vamos a trabajar.

print("--- FASE 1: Carga y Exploración de Datos ---")

# Intentamos cargar el archivo CSV. Si no lo encuentra, mostrará un error amigable.
try:
    df_original = pd.read_csv('Datos_de_prueba_expandidos.csv', encoding='utf-8')
    print("Archivo 'Datos_de_prueba_expandidos.csv' cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'Datos_de_prueba_expandidos.csv' no se encontró.")
    print("Asegúrate de que el script y el archivo CSV estén en el mismo directorio.")
    exit() # Si no hay datos, no podemos continuar.

# Mostramos las primeras 5 filas para tener una idea de cómo se ven los datos.
print("\nPrimeras 5 filas del dataset:")
print(df_original.head())

# Mostramos información general: número de filas, columnas y tipo de dato de cada una.
print("\nInformación general del dataset:")
df_original.info()


# =============================================================================
# FASE 2: PREPROCESAMIENTO DE DATOS
# =============================================================================
# Los modelos de Machine Learning son como calculadoras muy avanzadas: solo entienden números.
# No pueden trabajar directamente con texto ('Aceptada', 'Rechazada', 'USA', 'Mexico').
# En esta fase, "traducimos" nuestros datos a un lenguaje que el modelo pueda entender.

print("\n--- FASE 2: Preprocesamiento de Datos ---")

# --- Paso 2.1: Limpieza y Preparación General ---

# Para mantener nuestro DataFrame original intacto, creamos una copia para trabajar.
df = df_original.copy()

# Eliminamos columnas que son identificadores únicos o información personal.
# Estas columnas no ayudan al modelo a encontrar patrones y pueden confundirlo.
columnas_a_eliminar = ['Número de tarjeta de crédito', 'Nombre de la persona', 'Correo electrónico']
df = df.drop(columns=columnas_a_eliminar)
print(f"Columnas irrelevantes eliminadas: {columnas_a_eliminar}")

# --- Paso 2.2: Convertir Variables Categóricas a Numéricas ---
# Usaremos una técnica llamada "One-Hot Encoding".
# Analogía: Imagina que tienes una columna 'Color' con valores 'Rojo', 'Verde', 'Azul'.
# One-Hot Encoding la convierte en tres columnas: 'Color_Rojo', 'Color_Verde', 'Color_Azul'.
# Si una fila era 'Rojo', ahora tendrá un 1 en 'Color_Rojo' y 0 en las otras dos.
# Esto permite al modelo entender las categorías sin asumir un orden incorrecto entre ellas.

df_procesado = pd.get_dummies(df, drop_first=True)
# `drop_first=True` elimina una categoría de cada variable para evitar redundancia.
# Por ejemplo, si no es 'País_USA' y no es 'País_Canada', el modelo ya sabe que es el país restante.

print("\nVariables de texto (categóricas) convertidas a formato numérico.")

# Corregimos el nombre de la columna objetivo. pd.get_dummies con drop_first=True crea 'Resultado_Rechazada'.
# La renombramos a 'Resultado_Aceptada' y la invertimos (0 -> 1, 1 -> 0) para que 1 signifique 'Aceptada'.
df_procesado = df_procesado.rename(columns={'Resultado_Rechazada': 'Resultado_Aceptada'})
df_procesado['Resultado_Aceptada'] = 1 - df_procesado['Resultado_Aceptada']

print("Nuevas columnas creadas por One-Hot Encoding:")
print(df_procesado.head())


# =============================================================================
# TAREA 1: CLASIFICACIÓN (Predecir si una transacción es Aceptada o Rechazada)
# =============================================================================

print("\n" + "="*60)
print("INICIO DE LA TAREA DE CLASIFICACIÓN")
print("="*60)

# --- Paso 3.1: Separar Características (X) y Variable Objetivo (y) ---
# 'y' es lo que queremos predecir: 'Resultado_Aceptada' (1 si fue aceptada, 0 si no).
# 'X' es todo lo demás, las pistas que usaremos para hacer la predicción.

X_cls = df_procesado.drop('Resultado_Aceptada', axis=1)
y_cls = df_procesado['Resultado_Aceptada']

print("\nDatos separados para la tarea de CLASIFICACIÓN.")

# --- Paso 3.2: Dividir los datos en Entrenamiento y Prueba ---
# ¡Este es un paso CRÍTICO! Es como estudiar para un examen.
# - Set de Entrenamiento (70%): El material de estudio que le damos al modelo para que aprenda.
# - Set de Prueba (30%): Un examen sorpresa con preguntas que el modelo nunca ha visto.
# Esto nos permite evaluar si el modelo realmente "aprendió" a generalizar o si solo
# "memorizó" las respuestas del material de estudio (lo que se llama "sobreajuste" o "overfitting").

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)
# `test_size=0.3` significa que el 30% de los datos se usarán para la prueba.
# `random_state=42` es como elegir siempre el mismo número en un sorteo, para que la división sea reproducible.
# `stratify=y_cls` asegura que la proporción de 'Aceptadas' y 'Rechazadas' sea la misma en ambos sets.

print(f"Datos de clasificación divididos: {len(X_train_cls)} para entrenamiento y {len(X_test_cls)} para prueba.")

# --- Paso 3.3: Crear y Entrenar el Modelo de Clasificación XGBoost ---
print("\nEntrenando el modelo de clasificación XGBoost...")

# Creamos una instancia del clasificador de XGBoost.
# `use_label_encoder=False` y `eval_metric='logloss'` son configuraciones recomendadas para evitar advertencias.
modelo_clasificacion = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# ¡El entrenamiento! Aquí es donde el modelo "estudia" los datos de entrenamiento.
# Aprende los patrones que conectan las características (X_train_cls) con el resultado (y_train_cls).
modelo_clasificacion.fit(X_train_cls, y_train_cls)

print("Modelo de clasificación entrenado exitosamente.")

# --- Paso 3.4: Predicción y Evaluación del Clasificador ---
print("\n--- Evaluación del Modelo de Clasificación ---")

# Usamos el modelo entrenado para hacer predicciones sobre el "examen sorpresa" (los datos de prueba).
y_pred_cls = modelo_clasificacion.predict(X_test_cls)

# Medimos qué tan bien lo hizo.
# Accuracy (Exactitud): ¿Qué porcentaje de predicciones fueron correctas?
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"\nAccuracy (Exactitud) en el conjunto de prueba: {accuracy:.4f}")
print(f"Esto significa que el modelo acertó en el {accuracy:.2%} de las predicciones.")

# Reporte de Clasificación: Un análisis más profundo.
# - Precisión: De todas las veces que el modelo predijo 'Aceptada', ¿cuántas veces acertó?
# - Recall (Exhaustividad): De todas las transacciones que realmente eran 'Aceptada', ¿cuántas encontró el modelo?
# - F1-Score: Una media que combina Precisión y Recall.
print("\nReporte de Clasificación:")
print(classification_report(y_test_cls, y_pred_cls, target_names=['Rechazada (0)', 'Aceptada (1)']))


# =============================================================================
# TAREA 2: REGRESIÓN (Predecir el Monto de la Transacción)
# =============================================================================

print("\n" + "="*60)
print("INICIO DE LA TAREA DE REGRESIÓN")
print("="*60)

# --- Paso 4.1: Separar Características (X) y Variable Objetivo (y) ---
# Ahora, lo que queremos predecir ('y') es una cantidad numérica: 'Monto de la transacción'.
# Las características ('X') serán el resto de las columnas (incluyendo si la transacción fue aceptada o no).

# Nota: Para esta tarea, la columna 'Resultado_Aceptada' se convierte en una característica más.
X_reg = df_procesado.drop('Monto de transacción', axis=1)
y_reg = df_procesado['Monto de transacción']

print("\nDatos separados para la tarea de REGRESIÓN.")

# --- Paso 4.2: Dividir los datos en Entrenamiento y Prueba ---
# Usamos la misma lógica que antes: un set para estudiar y otro para el examen.

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
# Para regresión, 'stratify' no se usa porque la variable objetivo es continua (no tiene clases).

print(f"Datos de regresión divididos: {len(X_train_reg)} para entrenamiento y {len(X_test_reg)} para prueba.")

# --- Paso 4.3: Crear y Entrenar el Modelo de Regresión XGBoost ---
print("\nEntrenando el modelo de regresión XGBoost...")

# Creamos una instancia del regresor de XGBoost.
# `objective='reg:squarederror'` le dice a XGBoost que el objetivo es minimizar el error cuadrático,
# lo cual es estándar para tareas de regresión.
modelo_regresion = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Entrenamos el modelo para que aprenda a predecir el monto basándose en las características.
modelo_regresion.fit(X_train_reg, y_train_reg)

print("Modelo de regresión entrenado exitosamente.")

# --- Paso 4.4: Predicción y Evaluación del Regresor ---
print("\n--- Evaluación del Modelo de Regresión ---")

# Hacemos predicciones sobre el monto de la transacción para los datos de prueba.
y_pred_reg = modelo_regresion.predict(X_test_reg)

# En regresión, no medimos "aciertos", sino "qué tan cerca" estuvieron las predicciones de los valores reales.

# Mean Absolute Error (MAE): El promedio del error absoluto.
# Analogía: Si el modelo predijo $105 y el valor real era $100, el error absoluto es $5.
# MAE nos dice, en promedio, por cuántos dólares se equivocó el modelo en sus predicciones.
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"En promedio, las predicciones del modelo se desvían en ${mae:.2f} del monto real.")

# Root Mean Squared Error (RMSE): La raíz cuadrada del promedio de los errores al cuadrado.
# Es similar al MAE, pero penaliza más los errores grandes.
# Si el MAE es el error promedio, el RMSE te da una idea de la magnitud típica del error.
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Esta métrica también indica la desviación promedio, pero penaliza más los errores grandes.")

# --- Paso 4.5: Comparando Predicciones con Valores Reales ---
# Veamos algunos ejemplos para entender mejor el rendimiento del modelo.

df_comparacion = pd.DataFrame({
    'Monto Real': y_test_reg.values,
    'Monto Predicho': y_pred_reg.flatten()
})
df_comparacion['Diferencia'] = df_comparacion['Monto Real'] - df_comparacion['Monto Predicho']

print("\nComparación de algunos valores reales vs. predichos por el modelo de regresión:")
print(df_comparacion.head(10))

print("\n" + "="*60)
print("FIN DEL SCRIPT")
print("="*60)

"""
Conclusión:

¡Felicidades! Has implementado y evaluado dos tipos de modelos con XGBoost.

1. Clasificación: Aprendiste a predecir una categoría (Aceptada/Rechazada) y a evaluar
   el modelo con métricas como Accuracy y el Reporte de Clasificación.

2. Regresión: Aprendiste a predecir un valor numérico (Monto) y a evaluar el modelo
   con métricas como MAE y RMSE, que miden la magnitud del error.

Este script es solo el comienzo. XGBoost es una herramienta muy poderosa con muchos
parámetros que se pueden ajustar (lo que se conoce como "ajuste de hiperparámetros")
para mejorar aún más el rendimiento de los modelos.
"""
