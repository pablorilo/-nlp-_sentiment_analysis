ANÁLISIS DEL SENTIMIENTO DE RESEÑAS DE AMAZON

En este proyecto, se llevó a cabo el análisis del sentimiento en reseñas de Amazon sobre aparatos electrónicos. El proyecto se divide en cuatro partes bien diferenciadas:

1. Descarga de datos

En el notebook "download_ade.ipynb" se realiza la descarga del set de datos. Tras la descarga, se lleva a cabo una pequeña transformación de los datos a un formato que permita trabajar con ellos de forma sencilla. Además, se realiza un análisis exploratorio de la información que se utilizará como base para el preprocesado, la segunda parte del proyecto.

2. Preprocesado

En el notebook "preprocesing.ipynb" se generaron dos clases que se encargan de dos preprocesados diferentes. La primera realiza un preprocesado de los datos en bruto, eliminando signos de puntuación, realizando la lematización, generando las pos_tags y eliminando las stop_words. En la segunda clase se genera un pipeline que prepara los datos para realizar un modelo de CNN, generando una capa de embedding con W2V que posteriormente se usará en un modelo CNN con la capa de embedding customizada.

3. Entrenamiento

En el notebook "training.ipynb" se generaron dos clases que se encargan del entrenamiento de forma automática. La primera clase, "TextPipelineLogisticAndBoos", genera un pipeline con el método de extracción de características y el algoritmo de entrenamiento que se le pase por parámetro. En este punto, se generaron cuatro combinaciones de pipeline: TfidfVectorizer -> SelectKBest(chi2) -> LogisticRegression, CountVectorizer -> SelectKBest(chi2) -> LogisticRegression, TfidfVectorizer -> SelectKBest(chi2) -> GradientBoostingClassifier y CountVectorizer -> SelectKBest(chi2) -> GradientBoostingClassifier. La segunda clase, "TrainingCNN", realizó varios entrenamientos: primeramente, tres entrenamientos con la capa de embedding generada en la propia CNN ('LSTM', 'GRU', 'SimpleRNN'), y en la segunda parte se llevaron a cabo los entrenamientos con un embedding customizado con W2V.

4. Conclusiones

En el notebook "conclusion.ipynb" se presentan las gráficas con los resultados obtenidos en los diferentes análisis y se exponen las conclusiones obtenidas.


Además, el archivo "graphics.py" contiene una clase con diferentes métodos que imprimen gráficos en pantalla, importada automáticamente en cada notebook. Cada clase y método desarrollados en el proyecto cuenta con una breve información que nos ayuda a entender el proceso que siguen.