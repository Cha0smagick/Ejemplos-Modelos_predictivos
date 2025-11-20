# -*- coding: utf-8 -*-
"""
Script Didáctico de Gradient Boosting Classifier.

Objetivo:
- Implementar un modelo de clasificación usando GradientBoostingClassifier de scikit-learn.
- Predecir si una transacción será 'Aceptada' o 'Rechazada'.
- Explicar cada paso de forma detallada y con analogías para principiantes.

Analogía de Gradient Boosting:
Imagina que estás construyendo un mueble complejo en equipo.
1. El primer carpintero (un "árbol de decisión" débil) construye una versión muy simple y básica del mueble.
   Comete muchos errores evidentes.
2. El segundo carpintero no empieza de cero. En su lugar, observa los errores del primero (una pata torcida,
   un cajón que no cierra) y se enfoca exclusivamente en arreglar esos problemas.
3. El tercer carpintero observa los errores que quedaron después del trabajo del segundo y se enfoca en
   corregir esas imperfecciones más pequeñas.
4. El proceso continúa: cada nuevo carpintero que se une al equipo es un especialista en corregir los
   errores residuales de los anteriores.

Gradient Boosting funciona así: construye modelos (árboles) de forma secuencial, donde cada nuevo
árbol se entrena para corregir los errores del conjunto de árboles anterior. El resultado final es un
modelo muy preciso y potente, como un mueble finamente acabado por un equipo de expertos.

Librerías utilizadas:
- pandas: Para cargar y manipular los datos en tablas (DataFrames).
- scikit-learn: La principal biblioteca de Machine Learning en Python. Usaremos:
  - train_test_split: Para dividir los datos.
  - GradientBoostingClassifier: El algoritmo que implementaremos.
  - accuracy_score, confusion_matrix, classification_report: Para evaluar el modelo.
- numpy: Para operaciones numéricas.
"""

# =============================================================================
# FASE 0: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# Antes de empezar, reunimos todas las "herramientas" (librerías) que vamos a necesitar.
# Es como preparar los ingredientes y utensilios antes de cocinar.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# FASE 1: CARGA DE DATOS
# =============================================================================
# El primer paso es cargar nuestros datos y echarles un vistazo para entenderlos.
# Es como leer la lista de ingredientes y ver qué tenemos antes de empezar.

print("--- FASE 1: Carga de Datos ---")

# Definimos el nombre del archivo que contiene nuestros datos.
file_name = 'Datos_de_prueba_expandidos.csv'

# Intentamos cargar el archivo. Si no lo encuentra, mostrará un error amigable.
try:
    df_original = pd.read_csv(file_name, encoding='utf-8')
    print(f"Archivo '{file_name}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{file_name}' no se encontró.")
    print("Asegúrate de que el script y el archivo CSV estén en el mismo directorio.")
    exit() # Si no hay datos, no podemos continuar.


# =============================================================================
# FASE 2: PREPARACIÓN DE DATOS (ÉNFASIS EDUCATIVO)
# =============================================================================
# Los modelos de Machine Learning son como calculadoras muy avanzadas: solo entienden números.
# No pueden trabajar con texto ('Mexico', 'Aceptada') o datos irrelevantes.
# En esta fase, "limpiamos y traducimos" nuestros datos a un lenguaje que el modelo pueda entender.

print("\n--- FASE 2: Preparación de Datos ---")

# --- Paso 2.1: Limpieza y Selección de Datos ---
# Creamos una copia para no modificar el DataFrame original.
df = df_original.copy()

# Eliminamos columnas que no aportan información útil para la predicción, como identificadores únicos.
# Incluirlos podría confundir al modelo o llevarlo a memorizar en lugar de aprender.
columnas_a_eliminar = ['Número de tarjeta de crédito', 'Nombre de la persona', 'Correo electrónico', 'Fecha de transacción', 'Hora de transacción']
df = df.drop(columns=columnas_a_eliminar)
print(f"Columnas irrelevantes eliminadas: {columnas_a_eliminar}")

# --- Paso 2.2: Identificación de Características (X) y Objetivo (y) ---
# 'y' (el objetivo o "target"): Es lo que queremos predecir. En este caso, si la transacción fue 'Aceptada' o 'Rechazada'.
# 'X' (las características o "features"): Son todas las demás columnas que usaremos como pistas para hacer la predicción.

# La columna 'Resultado' es nuestro objetivo. La convertimos a números: 1 para 'Aceptada' y 0 para 'Rechazada'.
df['Resultado'] = df['Resultado'].map({'Aceptada': 1, 'Rechazada': 0})

X = df.drop('Resultado', axis=1) # X son todas las columnas EXCEPTO la de 'Resultado'.
y = df['Resultado']             # y es ÚNICAMENTE la columna 'Resultado'.

print("\nDatos separados en características (X) y variable objetivo (y).")

# --- Paso 2.3: Codificación de Variables Categóricas ---
# El modelo no entiende texto como 'USA' o 'Mexico'. Debemos convertirlo a números.
# Usaremos "One-Hot Encoding", que crea nuevas columnas para cada categoría.
# Por ejemplo, si 'País' tiene 'USA' y 'Mexico', se crearán 'País_USA' y 'País_Mexico'.
# Una fila con 'USA' tendrá un 1 en 'País_USA' y un 0 en 'País_Mexico'.
X_procesado = pd.get_dummies(X, drop_first=True)
print("Variables de texto (categóricas) convertidas a formato numérico con One-Hot Encoding.")
print("Vista de las características (X) después de la codificación:")
print(X_procesado.head())

# --- Paso 2.4: División de Datos (Entrenamiento y Prueba) ---
# ¡Este es un paso CRÍTICO para evaluar nuestro modelo de forma honesta!
# Analogía: Es como prepararse para un examen.
# - Conjunto de Entrenamiento (70%): El "libro de texto" que le damos al modelo para que aprenda patrones.
# - Conjunto de Prueba (30%): Un "examen sorpresa" con preguntas (datos) que el modelo nunca ha visto.
# Si al modelo le va bien en el examen, significa que "aprendió" a generalizar. Si solo le va bien
# con los datos que ya conocía, significa que "memorizó" (esto se llama "sobreajuste" u "overfitting").

X_train, X_test, y_train, y_test = train_test_split(
    X_procesado, y,
    test_size=0.3,      # 30% de los datos para el conjunto de prueba.
    random_state=42,    # Semilla para que la división sea siempre la misma y los resultados sean reproducibles.
    stratify=y          # Asegura que la proporción de 'Aceptadas' y 'Rechazadas' sea la misma en ambos conjuntos.
)

print(f"\nDatos divididos: {len(X_train)} filas para entrenamiento y {len(X_test)} para prueba.")


# =============================================================================
# FASE 3: IMPLEMENTACIÓN Y ENTRENAMIENTO DEL MODELO
# =============================================================================
# Ahora que los datos están listos, creamos y entrenamos a nuestro "equipo de expertos".

print("\n--- FASE 3: Entrenamiento del Modelo Gradient Boosting ---")

# --- Paso 3.1: Inicialización del Modelo ---
# Creamos una instancia del clasificador Gradient Boosting.
# n_estimators=100: Nuestro "equipo" tendrá 100 "expertos" (árboles) que aprenderán uno del otro.
# learning_rate=0.1: Controla qué tan rápido aprende cada nuevo árbol de los errores del anterior. Un valor bajo lo hace más robusto.
# random_state=42: Para que la aleatoriedad interna del modelo sea reproducible.
modelo_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# --- Paso 3.2: Ajuste (Entrenamiento) del Modelo ---
# Aquí es donde ocurre la "magia". El modelo "estudia" los datos de entrenamiento.
# La función .fit() es el proceso de aprendizaje, donde el modelo analiza X_train (las pistas)
# y y_train (las respuestas correctas) para encontrar los patrones que los conectan.
modelo_gb.fit(X_train, y_train)

print("Modelo GradientBoostingClassifier entrenado exitosamente.")


# =============================================================================
# FASE 4: EVALUACIÓN Y EXPLICACIÓN DEL RESULTADO
# =============================================================================
# ¿Qué tan bueno es nuestro modelo? Vamos a medir su rendimiento en el "examen sorpresa".

print("\n--- FASE 4: Evaluación del Modelo ---")

# --- Paso 4.1: Predicción ---
# Usamos el modelo ya entrenado para hacer predicciones sobre el conjunto de prueba.
y_pred = modelo_gb.predict(X_test)

# --- Paso 4.2: Métricas de Rendimiento ---

# Accuracy (Exactitud): ¿Qué porcentaje de predicciones fueron correctas?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Exactitud): {accuracy:.4f}")
print(f"Esto significa que el modelo acertó en el {accuracy:.2%} de las predicciones sobre datos nuevos.")

# --- Paso 4.3: Matriz de Confusión (Análisis de Errores) ---
# Esta matriz nos da un desglose detallado de los aciertos y errores del modelo.
print("\nMatriz de Confusión:")
matriz = confusion_matrix(y_test, y_pred)
print(matriz)
print("\nInterpretación de la Matriz de Confusión:")
print(f" - Verdaderos Negativos (TN): {matriz[0, 0]} -> El modelo predijo 'Rechazada' y acertó.")
print(f" - Falsos Positivos (FP):    {matriz[0, 1]} -> El modelo predijo 'Aceptada' pero era 'Rechazada'. (Error Tipo I)")
print(f" - Falsos Negativos (FN):    {matriz[1, 0]} -> El modelo predijo 'Rechazada' pero era 'Aceptada'. (Error Tipo II)")
print(f" - Verdaderos Positivos (TP): {matriz[1, 1]} -> El modelo predijo 'Aceptada' y acertó.")

# --- Paso 4.4: Reporte de Clasificación ---
# Muestra métricas clave como Precisión, Recall y F1-Score para cada clase.
# - Precisión: De los que predijo como 'Aceptada', ¿cuántos acertó?
# - Recall (Sensibilidad): De todos los que realmente eran 'Aceptada', ¿cuántos encontró?
# - F1-Score: Es una media armónica de Precisión y Recall, útil para clases desbalanceadas.
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, target_names=['Rechazada (0)', 'Aceptada (1)']))

# --- Paso 4.5: Importancia de las Características ---
# ¿En qué "pistas" (características) se fijó más nuestro modelo para tomar sus decisiones?
# Esto nos ayuda a entender qué variables son más influyentes.
print("\n--- Importancia de las Características ---")
print("Mostrando qué variables fueron más influyentes para las predicciones del modelo:")

# Creamos un DataFrame para visualizar la importancia de cada característica.
df_importancia = pd.DataFrame({
    'Variable': X_procesado.columns,
    'Importancia': modelo_gb.feature_importances_
}).sort_values('Importancia', ascending=False)

# Mostramos las 10 características más importantes.
print(df_importancia.head(10))

print("\n" + "="*60)
print("FIN DEL SCRIPT")
print("="*60)

