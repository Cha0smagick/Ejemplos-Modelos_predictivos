# -*- coding: utf-8 -*-
"""
Script Didáctico de K-Nearest Neighbors (kNN) para Clasificación.

Objetivo:
- Implementar un modelo predictivo usando el algoritmo kNN para clasificar si una
  transacción será 'Aceptada' o 'Rechazada'.
- Explicar cada paso de forma extremadamente detallada y con analogías, ideal para
  principiantes sin experiencia previa en programación o matemáticas.

Analogía de kNN:
Imagina que eres nuevo en un vecindario y quieres saber a qué equipo de fútbol apoyar.
Una forma de decidir sería mirar las camisetas de tus 5 vecinos más cercanos (K=5).
Si 3 de ellos usan la camiseta del equipo 'A' y 2 la del equipo 'B', lo más probable
es que tú también decidas apoyar al equipo 'A'.

kNN funciona exactamente así: para clasificar un nuevo punto de datos, mira a sus 'K'
vecinos más cercanos en los datos de entrenamiento y le asigna la clase que sea más
común entre ellos (la "votación" de los vecinos).

Librerías utilizadas:
- os: Para interactuar con el sistema operativo (verificar si un archivo existe).
- pandas: Para cargar y manipular los datos en tablas (DataFrames).
- numpy: Para operaciones numéricas y la creación de datos de ejemplo.
- scikit-learn: La principal biblioteca de Machine Learning en Python.
- matplotlib: Para crear gráficos y visualizaciones.
"""

# =============================================================================
# FASE 0: PREPARACIÓN E IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# Antes de empezar, reunimos todas las "herramientas" (librerías) que vamos a necesitar.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- Bloque de Simulación de Datos ---
# Verificamos si el archivo de datos ya existe. Si no, creamos uno de ejemplo.
# Esto asegura que el script siempre pueda ejecutarse, incluso si no tienes el archivo.
file_name = 'Datos_de_prueba_expandidos.csv'
if not os.path.exists(file_name):
    print(f"El archivo '{file_name}' no se encontró. Creando datos de ejemplo...")
    # Usamos una "semilla" para que los datos aleatorios sean siempre los mismos.
    np.random.seed(42)
    num_filas = 100
    datos = {
        'Monto de transacción': np.random.uniform(10, 5000, size=num_filas).round(2),
        'Edad_Cliente': np.random.randint(18, 75, size=num_filas),
        'Puntuacion_Crediticia': np.random.randint(300, 850, size=num_filas),
        'Resultado': np.random.choice(['Aceptada', 'Rechazada'], size=num_filas, p=[0.6, 0.4])
    }
    df = pd.DataFrame(datos)
    print(f"Archivo de ejemplo creado con {num_filas} filas.")
else:
    # Si el archivo existe, lo cargamos.
    # COMENTARIO PARA EL USUARIO: Si tu archivo tiene otro nombre, cámbialo aquí.
    df = pd.read_csv(file_name, encoding='utf-8')
    print(f"Archivo '{file_name}' cargado exitosamente.")


# =============================================================================
# FASE 1: CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================
# El primer paso es cargar nuestros datos y echarles un vistazo para entenderlos.

print("\n--- FASE 1: Exploración de Datos ---")

# Un DataFrame es, en esencia, una tabla como las que verías en Excel.
# .head() nos muestra las primeras filas para tener una idea de cómo se ven los datos.
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# .info() nos da un resumen técnico: número de filas, columnas y el tipo de dato de cada una.
# Es crucial para identificar qué columnas son numéricas y cuáles son texto (categóricas).
print("\nInformación general del dataset:")
df.info()


# =============================================================================
# FASE 2: PREPARACIÓN DE DATOS (EL ALMA DE KNN)
# =============================================================================
# kNN es muy sensible a cómo se preparan los datos. Esta fase es la más importante.

print("\n--- FASE 2: Preparación de Datos ---")

# --- Paso 2.1: Codificar y Seleccionar Características y Objetivo ---
# Para este ejemplo didáctico, usaremos 'Monto de transacción' y 'País de emisión'.
# Como 'País de emisión' es texto, primero lo convertimos a números usando LabelEncoder.
# Esto asignará un número único a cada país (ej. USA=5, Canada=1, etc.).
le_pais = LabelEncoder()
df['Pais_codificado'] = le_pais.fit_transform(df['País de emisión'])

features = ['Monto de transacción', 'Pais_codificado']
target = 'Resultado'

# Nos aseguramos de que no haya filas con valores faltantes en las columnas que usaremos.
# Esto es crucial para que el modelo funcione correctamente.
# .dropna() elimina las filas que tengan al menos un valor nulo en el subconjunto de columnas especificado.
df_modelo = df[features + [target]].dropna()

# 'X' son las características (las pistas).
# 'y' es el objetivo (lo que queremos predecir).
X = df_modelo[features]
y_texto = df_modelo[target]

# Convertimos la variable objetivo de texto ('Aceptada'/'Rechazada') a números (1/0).
le = LabelEncoder()
y = le.fit_transform(y_texto)

print("\nDatos separados en características (X) y variable objetivo (y).")

# --- Paso 2.2: Estandarización de Datos (¡CRÍTICO PARA KNN!) ---
# ¿Por qué es tan importante? kNN se basa en medir distancias.
# Imagina que una característica es 'Salario' (ej. 50000) y otra es 'Edad' (ej. 30).
# Si no las estandarizamos, el salario dominará completamente el cálculo de la distancia,
# y la edad apenas tendrá impacto, lo cual es incorrecto.
# StandardScaler transforma los datos para que todas las características tengan una media de 0
# y una desviación estándar de 1, poniéndolas en la misma "escala".

scaler = StandardScaler()
X_estandarizado = scaler.fit_transform(X)

print("Características estandarizadas para que todas tengan la misma importancia en el cálculo de distancia.")

# --- Paso 2.3: División en Entrenamiento y Prueba ---
# Dividimos los datos para simular un "examen".
# - Conjunto de Entrenamiento: Los datos que el modelo "estudia".
# - Conjunto de Prueba: Los datos que el modelo nunca ha visto, usados para evaluarlo honestamente.
X_train, X_test, y_train, y_test = train_test_split(
    X_estandarizado, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Datos divididos: {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")


# =============================================================================
# FASE 3: EL CONCEPTO MATEMÁTICO DE DISTANCIA
# =============================================================================
# kNN usa una fórmula para medir la "cercanía" entre puntos. La más común es la Distancia Euclidiana.
# Analogía: Es como calcular la distancia en línea recta entre dos ciudades en un mapa.
#
# Fórmula para 2 dimensiones (ej. Monto y Puntuación):
# d = √((monto₂ - monto₁)² + (puntuacion₂ - puntuacion₁)²)
#
# El algoritmo extiende esta fórmula a todas las dimensiones (características) que tengamos.
print("\n--- FASE 3: Entendiendo la Distancia Euclidiana (Base de kNN) ---")
print("kNN calcula la 'cercanía' usando una fórmula de distancia, como la euclidiana.")


# =============================================================================
# FASE 4: ENTRENAMIENTO Y PREDICCIÓN (LA VOTACIÓN)
# =============================================================================

print("\n--- FASE 4: Entrenamiento y Predicción ---")

# --- Paso 4.1: "Entrenamiento" del Modelo ---
# El "entrenamiento" en kNN es muy simple: solo memoriza la ubicación de todos los puntos de datos de entrenamiento.
# No aprende una fórmula compleja como otros modelos.
# K=5: Le decimos al modelo que considere a los 5 vecinos más cercanos para cada predicción.
k = 5
modelo_knn = KNeighborsClassifier(n_neighbors=k)
modelo_knn.fit(X_train, y_train)

print(f"Modelo kNN 'entrenado' con K={k}. El modelo ha memorizado los datos de entrenamiento.")

# --- Paso 4.2: Predicción ---
# Para cada punto en el conjunto de prueba (X_test), el modelo:
# 1. Calcula la distancia a TODOS los puntos del conjunto de entrenamiento (X_train).
# 2. Encuentra los 'K' puntos más cercanos.
# 3. Realiza una "votación": la clase más común entre esos 'K' vecinos es la predicción final.
y_pred = modelo_knn.predict(X_test)
print("Predicciones realizadas sobre el conjunto de prueba mediante la 'votación' de los vecinos.")


# =============================================================================
# FASE 5: EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

print("\n--- FASE 5: Evaluación y Visualización ---")

# --- Paso 5.1: Métrica de Precisión (Accuracy) ---
# Comparamos las predicciones del modelo (y_pred) con las respuestas correctas (y_test).
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Exactitud) del modelo: {accuracy:.2%}")
print("Esto significa que el modelo clasificó correctamente ese porcentaje de las transacciones de prueba.")

# --- Paso 5.2: Reporte de Clasificación ---
# Ofrece una visión más completa del rendimiento.
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Paso 5.3: Visualización del Concepto kNN ---
print("\nGenerando gráfico para visualizar el concepto de kNN...")

plt.figure(figsize=(10, 7))
# Graficamos los puntos de entrenamiento
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, label='Datos de Entrenamiento')

# Simulamos un nuevo punto que queremos clasificar
nuevo_punto = np.array([[-0.5, 0.5]]) # Un punto de ejemplo en el espacio estandarizado
plt.scatter(nuevo_punto[:, 0], nuevo_punto[:, 1], c='green', marker='X', s=150, label='Nuevo Punto a Clasificar')

# Encontramos y graficamos los K vecinos más cercanos a este nuevo punto
distancias, indices = modelo_knn.kneighbors(nuevo_punto)
vecinos_cercanos = X_train[indices[0]]
plt.scatter(vecinos_cercanos[:, 0], vecinos_cercanos[:, 1], facecolors='none', edgecolors='green', s=200, label=f'{k} Vecinos Más Cercanos')

# Añadimos títulos y etiquetas
plt.title('Visualización del Concepto K-Nearest Neighbors (kNN)')
plt.xlabel(f'Característica 1: {features[0]} (Estandarizada)')
plt.ylabel(f'Característica 2: {features[1]} (Estandarizada)')

# Creamos una leyenda para las clases
handles, labels = scatter.legend_elements()
class_labels = le.classes_
legend1 = plt.legend(handles, class_labels, title="Clases")
plt.gca().add_artist(legend1) # Añadimos la primera leyenda

# Añadimos la leyenda para los otros elementos del gráfico
plt.legend(loc='upper right')

plt.grid(True)
plt.show()

print("\nGráfico mostrado. El 'Nuevo Punto' (X verde) se clasificaría según la clase mayoritaria de sus 5 vecinos más cercanos (círculos verdes).")
print("\n" + "="*60)
print("FIN DEL SCRIPT")
print("="*60)
