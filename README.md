# Proyecto Didáctico de Modelos Predictivos para IA

¡Bienvenido/a! Este repositorio está diseñado como un recurso educativo para estudiantes universitarios de especialización en Inteligencia Artificial que deseen aprender sobre la implementación práctica de modelos predictivos.

## 🎯 Objetivo del Proyecto

El objetivo principal es desmitificar los modelos de Machine Learning y la ciencia de datos a través de ejemplos prácticos, claros y extensamente comentados. Cada script en este repositorio es una guía paso a paso que no solo muestra *cómo* implementar un modelo, sino también *por qué* funciona y *qué* significan sus resultados.

La filosofía es "aprender haciendo", enfocándonos en:
- **Código Funcional:** Scripts completos que puedes ejecutar desde el principio hasta el fin.
- **Explicaciones Conceptuales:** Comentarios detallados que explican la teoría matemática y estadística de una manera intuitiva.
- **Aplicación Real:** Uso de datasets para resolver problemas concretos, como la predicción de series temporales.

## 📂 Estructura del Repositorio

Cada modelo predictivo se encuentra en su propio archivo de Python (`.py`). Los scripts están divididos en fases lógicas para facilitar el aprendizaje:

1.  **Fase 0: Preparación:** Importación de las librerías necesarias.
2.  **Fase 1: Carga y Preparación de Datos:** El paso crítico de limpiar y transformar los datos al formato que el modelo requiere.
3.  **Fase 2: Entendiendo el Modelo:** Una explicación conceptual y matemática (simplificada) de cómo funciona el modelo.
4.  **Fase 3: Entrenamiento:** El proceso de "enseñar" al modelo usando los datos históricos.
5.  **Fase 4: Predicción/Inferencia:** Usar el modelo entrenado para hacer predicciones sobre datos nuevos o futuros.
6.  **Fase 5: Visualización y Evaluación:** Interpretar los resultados a través de gráficos y métricas.

---

## 🧠 Modelos Incluidos

A continuación se presenta un resumen de los modelos disponibles en este proyecto.

### 1. Prophet de Meta (Series Temporales)

*   **Archivo:** `prophet-main.py` | **Tipo:** Series Temporales
*   **Tipo:** Modelo de Series Temporales.

#### ¿Qué es Prophet?

Prophet es un modelo desarrollado por Meta (Facebook) específicamente diseñado para predecir datos de series temporales. Es especialmente potente cuando los datos tienen **patrones estacionales fuertes** (por ejemplo, semanales, anuales) y efectos de días festivos. Su popularidad radica en que es fácil de usar, robusto ante datos faltantes y atípicos, y altamente interpretable.

#### ¿Cómo funciona?

Prophet utiliza un **modelo aditivo**, lo que significa que descompone la serie temporal en varios componentes y los suma para generar la predicción final. La fórmula principal es:

`y(t) = g(t) + s(t) + h(t) + e(t)`

Donde:
- **`y(t)`**: Es la predicción final en el tiempo `t`.
- **`g(t)` - Tendencia (Trend)**: Modela el cambio no periódico a lo largo del tiempo. Es la dirección general de los datos (¿crecen o decrecen a largo plazo?).
- **`s(t)` - Estacionalidad (Seasonality)**: Captura los patrones periódicos. Por ejemplo, el aumento de ventas cada fin de semana (estacionalidad semanal) o durante el verano (estacionalidad anual). Prophet utiliza series de Fourier para modelar estos ciclos.
- **`h(t)` - Festivos (Holidays)**: Modela el efecto de eventos irregulares pero predecibles, como Navidad, el Buen Fin o un feriado nacional, que impactan el comportamiento normal de los datos.
- **`e(t)` - Error**: Representa el ruido o las variaciones aleatorias que el modelo no puede explicar con los otros componentes.

La gran ventaja de este enfoque es que podemos visualizar cada componente por separado, lo que nos permite entender *por qué* el modelo hace una determinada predicción.

#### Aplicación en el Ejemplo

En el script `prophet-main.py`, utilizamos este modelo para un caso práctico de negocio:

1.  **Datos:** Se utiliza un archivo CSV con transacciones de ventas, que incluye la fecha y el monto de cada transacción.
2.  **Objetivo:** Predecir el **monto total de transacciones por día** para los próximos 365 días.
3.  **Proceso:**
    - Los datos de transacciones individuales se agrupan para obtener una suma total por día, creando así una serie temporal diaria.
    - Se prepara el DataFrame para que tenga las columnas requeridas por Prophet: `ds` (fecha) y `y` (valor a predecir).
    - Se entrena el modelo Prophet con los datos históricos.
    - Se genera un pronóstico a un año.
    - Finalmente, se visualiza tanto la predicción completa (con sus intervalos de incertidumbre) como los componentes individuales (tendencia, estacionalidad semanal y anual) que el modelo ha aprendido.

Este ejemplo te permitirá entender cómo pasar de datos brutos a un pronóstico accionable y, lo más importante, cómo interpretar los patrones que el modelo descubre.

### 2. Regresión Logística

*   **Archivo:** `regresion-logistica.py` | **Tipo:** Clasificación Binaria

#### ¿Qué es la Regresión Logística?

Es uno de los modelos fundamentales para problemas de **clasificación binaria** (predecir un resultado con solo dos opciones, como Sí/No, 1/0, Aceptada/Rechazada). A pesar de su nombre, se usa para clasificación, no para regresión. Es rápido, eficiente y, lo más importante, sus resultados son muy fáciles de interpretar.

#### ¿Cómo funciona?

La Regresión Logística calcula la **probabilidad** de que una observación pertenezca a una clase. Utiliza la **función sigmoide**, una curva en forma de "S" que transforma cualquier valor numérico en una probabilidad entre 0 y 1.

El modelo aprende un "coeficiente" (o peso) para cada variable de entrada.
- Un **coeficiente positivo** significa que al aumentar esa variable, aumenta la probabilidad del resultado "1" (ej. 'Aceptada').
- Un **coeficiente negativo** significa que al aumentar esa variable, disminuye la probabilidad del resultado "1".

Si la probabilidad calculada es mayor que un umbral (normalmente 0.5), el modelo predice la clase "1"; de lo contrario, predice "0".

#### Aplicación en el Ejemplo

En `regresion-logistica.py`, el modelo predice si una transacción será **'Aceptada' (1) o 'Rechazada' (0)**. El script te guía a través de:
1.  **Preprocesamiento:** Convertir variables de texto (como el país) en un formato numérico que el modelo entienda (usando *One-Hot Encoding*) y escalar las variables numéricas.
2.  **Entrenamiento:** El modelo aprende los coeficientes para variables como 'Monto de transacción', 'Puntuacion_Crediticia', etc.
3.  **Interpretación:** Se analizan los coeficientes para entender qué factores tienen el mayor impacto positivo o negativo en la probabilidad de que una transacción sea aceptada.

### 3. K-Nearest Neighbors (kNN)

*   **Archivo:** `k-nearest.py` | **Tipo:** Clasificación

#### ¿Qué es kNN?

Es un algoritmo de "aprendizaje perezoso" o basado en instancias. Su lógica es increíblemente intuitiva: "dime quiénes son tus vecinos y te diré quién eres". Para clasificar un nuevo dato, simplemente mira a los 'k' datos más cercanos (vecinos) en el conjunto de entrenamiento y le asigna la clase más común entre ellos.

#### ¿Cómo funciona?

1.  **Almacenamiento:** El "entrenamiento" de kNN consiste únicamente en memorizar todos los datos de entrenamiento y sus etiquetas.
2.  **Cálculo de Distancia:** Cuando llega un nuevo punto, calcula la distancia (generalmente la distancia euclidiana) entre este nuevo punto y **todos** los puntos del conjunto de entrenamiento.
3.  **Votación:** Identifica los 'k' puntos más cercanos (los vecinos con la menor distancia).
4.  **Clasificación:** La clase que más se repite entre esos 'k' vecinos es la predicción final para el nuevo punto.

**¡Importante!** kNN es muy sensible a la escala de las variables. Por eso, es crucial **estandarizar** los datos antes de entrenar el modelo, como se muestra en el script.

#### Aplicación en el Ejemplo

En `k-nearest.py`, el modelo clasifica una transacción como **'Aceptada' o 'Rechazada'**. El script se enfoca en:
1.  **Estandarización:** Demostrar por qué es vital escalar características como 'Monto de transacción' y 'País' para que ambas tengan la misma importancia en el cálculo de la distancia.
2.  **Visualización:** Se incluye un gráfico que ilustra visualmente el concepto: muestra un nuevo punto, identifica a sus 'k' vecinos más cercanos y cómo estos "votan" para decidir su clase.

### 4. Random Forest (Bosque Aleatorio)

*   **Archivo:** `random-forest.py` | **Tipo:** Clasificación (y Regresión)

#### ¿Qué es Random Forest?

Es un modelo de "aprendizaje en conjunto" (ensemble) que combina las predicciones de muchos **árboles de decisión** individuales para obtener una predicción final más robusta y precisa. La analogía es como consultar a un comité de expertos en lugar de a uno solo: la decisión del grupo suele ser mejor que la de cualquier individuo.

#### ¿Cómo funciona?

1.  **Bootstrap Aggregating (Bagging):** Crea múltiples subconjuntos de datos de entrenamiento tomando muestras aleatorias con reemplazo.
2.  **Construcción de Árboles:** Entrena un árbol de decisión en cada uno de estos subconjuntos. Para cada división en el árbol, solo considera un subconjunto aleatorio de las características disponibles. Esta doble aleatoriedad (en datos y en características) es lo que hace que los árboles sean diferentes entre sí y reduce el sobreajuste.
3.  **Votación:** Para una nueva predicción, cada árbol en el bosque emite su "voto". La clase que recibe la mayoría de los votos es la predicción final del bosque.

#### Aplicación en el Ejemplo

En `random-forest.py`, se utiliza un conjunto de datos sintéticos para predecir si un cliente realizará una **'Compra' (1) o 'No Compra' (0)**. El script muestra cómo:
1.  Generar datos de ejemplo realistas.
2.  Entrenar un bosque con 100 árboles (`n_estimators=100`).
3.  Evaluar el modelo usando una matriz de confusión para entender los tipos de errores (Falsos Positivos y Falsos Negativos).

### 5. Modelos de Gradient Boosting (Gradient Boosting, XGBoost, LightGBM)

Estos tres modelos pertenecen a la misma familia de algoritmos de "boosting", que son de los más potentes y populares en competencias de Machine Learning. La idea central es construir modelos de forma secuencial, donde cada nuevo modelo se enfoca en corregir los errores del anterior.

#### ¿Cómo funcionan?

Imagina un equipo de especialistas construyendo algo:
1.  El primer modelo (un árbol simple) hace una predicción inicial, cometiendo errores evidentes.
2.  El segundo modelo no predice el objetivo original, sino que se entrena para **predecir los errores** (residuos) del primer modelo.
3.  La predicción del segundo modelo se suma a la del primero, corrigiendo parte del error.
4.  Un tercer modelo se entrena para corregir los errores restantes, y así sucesivamente.

El resultado es un conjunto de modelos que, en equipo, logran una predicción extremadamente precisa.

---

#### 5.1. Gradient Boosting Classifier

*   **Archivo:** `Gradient-boosting-classifier.py` | **Tipo:** Clasificación
*   **Descripción:** Es la implementación fundamental de este concepto disponible en `scikit-learn`. Es robusta y una excelente base para entender el boosting. En el script, se usa para predecir si una transacción es **'Aceptada' o 'Rechazada'** y se muestra cómo analizar la "importancia de las características" para ver qué variables influyen más en la decisión.

---

#### 5.2. XGBoost (Extreme Gradient Boosting)

*   **Archivo:** `XGBoost.py` | **Tipo:** Clasificación y Regresión
*   **Descripción:** XGBoost es una implementación optimizada y de alto rendimiento del Gradient Boosting. Es famoso por su velocidad y precisión. Incluye mejoras como la regularización (para evitar el sobreajuste) y la capacidad de manejar valores faltantes de forma nativa.
*   **Aplicación en el Ejemplo:** El script `XGBoost.py` es único porque aborda **dos tareas**:
    1.  **Clasificación:** Predecir si una transacción es **'Aceptada' o 'Rechazada'**.
    2.  **Regresión:** Predecir el **'Monto de la transacción'** (un valor numérico continuo).
    Esto te permite comparar cómo se aplica el mismo algoritmo a dos tipos de problemas diferentes y cómo se evalúan (Accuracy para clasificación, MAE/RMSE para regresión).

---

#### 5.3. LightGBM (Light Gradient Boosting Machine)

*   **Archivo:** `LightGBM.py` | **Tipo:** Clasificación
*   **Descripción:** Es otra implementación de alto rendimiento, desarrollada por Microsoft. Su principal ventaja es la **velocidad y eficiencia con grandes conjuntos de datos**. A diferencia de otros árboles que crecen nivel por nivel (horizontalmente), LightGBM crece hoja por hoja (verticalmente), enfocándose en las hojas donde puede reducir más el error.
*   **Aplicación en el Ejemplo:** En `LightGBM.py`, se usa para un problema de clasificación de **'Aceptada'/'Rechazada'**. El script destaca cómo LightGBM puede manejar variables categóricas de forma nativa y eficiente, y cómo visualizar la importancia de las características para interpretar el modelo.

---

## 🚀 ¿Cómo empezar?

1.  Clona este repositorio en tu máquina local.
2.  Asegúrate de tener Python y crea un entorno virtual para instalar las librerías necesarias. Puedes instalarlas todas con pip:
    ```bash
    pip install pandas prophet matplotlib scikit-learn numpy seaborn lightgbm xgboost
    ```
3.  Abre el script del modelo que te interese (ej. `random-forest.py`) en tu editor de código o IDE favorito.
4.  Lee los comentarios y ejecuta el script celda por celda o todo de una vez para ver los resultados.
5.  ¡Experimenta! Cambia los parámetros, usa tus propios datos o intenta añadir nuevas funcionalidades como los días festivos (`add_country_holidays`).

¡Feliz aprendizaje!
