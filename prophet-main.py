# -*- coding: utf-8 -*-
"""
Script Didáctico del Modelo de Series Temporales Prophet.

Rol: Experto en ciencia de datos y educador.

Objetivo Principal:
- Crear un script de Python completo y funcional que sirva como material didáctico
  para estudiantes sin experiencia previa en programación, matemáticas o estadística.
- Implementar el modelo Prophet de Meta para realizar una predicción de series temporales
  sobre el total de montos de transacción diarios.

Tipo: Series Temporales.
Uso principal: Predecir datos de series temporales que tienen una fuerte estacionalidad
y efectos de vacaciones, como datos de ventas o tráfico web. Es un modelo desarrollado por Facebook (Meta).
En Python: Requiere la librería `prophet`.
Ventaja: Es altamente configurable e intuitivo, y maneja de forma automática valores atípicos y datos faltantes.
"""

# =============================================================================
# FASE 0: PREPARACIÓN E IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# Antes de empezar, reunimos todas las "herramientas" (librerías) que vamos a necesitar.

import pandas as pd  # Para cargar y manipular los datos en tablas (DataFrames).
from prophet import Prophet  # La librería principal para el modelo Prophet.
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones.

print("--- FASE 0: Librerías importadas correctamente ---")


# =============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS (¡CRÍTICO PARA PROPHET!)
# =============================================================================
# El primer paso es cargar nuestros datos y transformarlos al formato específico que Prophet necesita.

print("\n--- FASE 1: Carga y Preparación de Datos ---")

# Cargamos el archivo CSV en un DataFrame de pandas.
# Un DataFrame es, en esencia, una tabla como las que verías en Excel.
try:
    df_original = pd.read_csv('Datos_de_prueba_expandidos.csv', encoding='utf-8')
    print("Archivo 'Datos_de_prueba_expandidos.csv' cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'Datos_de_prueba_expandidos.csv' no se encontró.")
    print("Asegúrate de que el script y el archivo CSV estén en el mismo directorio.")
    exit() # Si no hay datos, no podemos continuar.

# --- Paso 1.1: Formateo de datos para Prophet ---
# Prophet es muy estricto con el formato de entrada. Necesita un DataFrame con dos columnas obligatorias:
# 1. 'ds': Debe contener las fechas (datestamps).
# 2. 'y': Debe contener el valor numérico que queremos predecir.

print("\nTransformando los datos al formato requerido por Prophet ('ds' y 'y')...")

# Seleccionamos solo las columnas que nos interesan.
df_prophet = df_original[['Fecha de transacción', 'Monto de transacción']].copy()

# Renombramos las columnas a 'ds' y 'y'.
df_prophet.rename(columns={'Fecha de transacción': 'ds', 'Monto de transacción': 'y'}, inplace=True)

# Convertimos la columna 'ds' a un formato de fecha y hora que Python pueda entender.
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# --- Paso 1.2: Agregación de datos por día ---
# Nuestros datos tienen múltiples transacciones por día. Para una serie temporal,
# necesitamos un único valor por cada punto en el tiempo (en este caso, por día).
# Agruparemos los datos por fecha ('ds') y sumaremos los montos ('y') para obtener el total diario.
print("Agregando montos de transacción para obtener un total por día...")
df_prophet_diario = df_prophet.groupby('ds').sum().reset_index()

print("\nPrimeras 5 filas del dataset listo para Prophet (datos diarios):")
print(df_prophet_diario.head())


# =============================================================================
# FASE 2: EL MODELO MATEMÁTICO DE PROPHET (ENFOQUE EXPLICATIVO)
# =============================================================================
# Prophet utiliza un modelo aditivo, lo que significa que descompone la serie temporal
# en varios componentes y simplemente los suma.
#
# La fórmula es: y(t) = g(t) + s(t) + h(t) + e(t)
#
# Analogía: Piensa en descomponer una canción en sus partes:
# - y(t): La canción completa (nuestros datos).
# - g(t) - Tendencia (Trend): Es la melodía principal de la canción, la dirección general
#   de los datos a lo largo del tiempo (¿están subiendo, bajando o se mantienen estables?).
#   Prophet puede modelar esto con una línea recta (crecimiento lineal) o una curva (crecimiento logístico).
#
# - s(t) - Estacionalidad (Seasonality): Es el ritmo o los patrones que se repiten.
#   Por ejemplo, las ventas pueden ser más altas los fines de semana (estacionalidad semanal)
#   o en diciembre (estacionalidad anual). Prophet usa una técnica matemática llamada
#   "series de Fourier" para capturar estos ciclos, que se puede pensar como combinar
#   varias ondas suaves para imitar las subidas y bajadas periódicas de los datos.
#
# - h(t) - Festivos (Holidays): Son los "solos de guitarra" o eventos especiales que no
#   siguen un ciclo regular pero que afectan a los datos, como Navidad, el Buen Fin o un feriado nacional.
#   Prophet permite añadir listas de estos días para mejorar la precisión.
#
# - e(t) - Error: Es el ruido de fondo o las variaciones impredecibles que el modelo no puede
#   explicar con los otros componentes.

print("\n--- FASE 2: Entendiendo el Modelo Aditivo de Prophet ---")
print("Prophet descompone los datos en Tendencia, Estacionalidad y Festivos.")


# =============================================================================
# FASE 3: CREACIÓN Y ENTRENAMIENTO DEL MODELO
# =============================================================================
# Ahora que los datos están listos, creamos una instancia del modelo y lo "entrenamos".

print("\n--- FASE 3: Creación y Entrenamiento del Modelo ---")

# Creamos una instancia del modelo Prophet.
# Por defecto, Prophet buscará estacionalidades anuales y semanales si los datos lo permiten.
modelo = Prophet()

# Entrenamos el modelo usando el método .fit() con nuestro DataFrame diario.
# En este paso, Prophet está "aprendiendo" los componentes (tendencia, estacionalidad, etc.)
# de nuestros datos históricos.
modelo.fit(df_prophet_diario)

print("Modelo Prophet entrenado exitosamente.")


# =============================================================================
# FASE 4: REALIZACIÓN DE PREDICCIONES A FUTURO
# =============================================================================
# Con el modelo entrenado, ahora podemos pedirle que prediga el futuro.

print("\n--- FASE 4: Realización de Predicciones a Futuro ---")

# Primero, necesitamos crear un "esqueleto" de fechas futuras para las que queremos una predicción.
# Usaremos el método `make_future_dataframe` para extender nuestro historial de fechas 365 días hacia el futuro.
futuro = modelo.make_future_dataframe(periods=365)

print(f"\nÚltimas fechas de nuestros datos históricos: \n{df_prophet_diario['ds'].tail(3).to_string(index=False)}")
print(f"Últimas fechas del DataFrame futuro (predicción a 1 año): \n{futuro['ds'].tail(3).to_string(index=False)}")

# Ahora, usamos el método .predict() sobre este DataFrame futuro para generar el pronóstico.
pronostico = modelo.predict(futuro)

print("\nPredicciones generadas. Mostrando las últimas filas del pronóstico:")
# Mostramos las últimas filas del DataFrame de predicción.
# - 'yhat': Es la predicción del modelo.
# - 'yhat_lower' y 'yhat_upper': Son los límites inferior y superior del "intervalo de confianza".
#   Prophet nos dice que, aunque su mejor predicción es 'yhat', es muy probable que el valor real
#   caiga dentro de este rango. Es una medida de la incertidumbre del modelo.
print(pronostico[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# =============================================================================
# FASE 5: VISUALIZACIÓN Y EVALUACIÓN DE RESULTADOS
# =============================================================================
# La mejor forma de entender un pronóstico de series temporales es visualmente.

print("\n--- FASE 5: Visualización de Resultados ---")

# --- Paso 5.1: Visualización del Pronóstico Completo ---
# Prophet tiene una función de ploteo muy conveniente.
print("Generando gráfico del pronóstico...")
fig1 = modelo.plot(pronostico)
plt.title('Pronóstico del Monto Total de Transacciones Diarias')
plt.xlabel('Fecha')
plt.ylabel('Monto Total Diario')
plt.show()
print("Gráfico mostrado. Interpretación:")
print("- Puntos negros: Son los datos reales históricos.")
print("- Línea azul ('yhat'): Es la predicción del modelo.")
print("- Área azul claro: Es el intervalo de incertidumbre. El modelo espera que los valores futuros caigan dentro de esta zona.")

# --- Paso 5.2: Visualización de los Componentes del Modelo ---
# Esta es una de las herramientas más poderosas de Prophet para entender CÓMO piensa el modelo.
print("\nGenerando gráficos de los componentes del modelo (tendencia y estacionalidades)...")
fig2 = modelo.plot_components(pronostico)
plt.show()
print("Gráficos de componentes mostrados. Interpretación:")
print("- Gráfico 'trend': Muestra la tendencia a largo plazo que el modelo ha detectado.")
print("- Gráfico 'weekly': Muestra el patrón semanal. Puedes ver qué días de la semana tienden a tener más o menos transacciones.")
print("- Gráfico 'yearly': Muestra el patrón anual. Puedes ver en qué épocas del año las transacciones suben o bajan.")


# =============================================================================
# FASE 6: CONCLUSIÓN
# =============================================================================
print("\n" + "="*60)
print("FIN DEL SCRIPT Y CONCLUSIONES")
print("="*60)
print("""
¡Felicidades! Has completado un ciclo completo de pronóstico con Prophet.

Lo que has aprendido:
1. Preparación de Datos: La importancia de formatear los datos con columnas 'ds' y 'y',
   y la necesidad de agregar los datos a una frecuencia constante (diaria en este caso).

2. Modelo Aditivo: Entendiste conceptualmente cómo Prophet descompone una serie temporal
   en tendencia y estacionalidad para hacer sus predicciones.

3. Predicción y Visualización: No solo generaste un pronóstico a un año, sino que también
   visualizaste la predicción junto con su intervalo de incertidumbre.

4. Interpretabilidad: Usaste `plot_components` para "abrir la caja negra" y ver los patrones
   específicos (semanales y anuales) que el modelo aprendió de los datos.

Siguiente Paso Sugerido:
- Para mejorar la precisión, podrías investigar cómo añadir festivos específicos de tu país
  usando el método `add_country_holidays`. Por ejemplo, para México, sería:
  `modelo.add_country_holidays(country_name='MX')` antes de entrenar el modelo.
""")
