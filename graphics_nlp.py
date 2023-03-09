import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from wordcloud import WordCloud

import pandas as pd
import numpy as np

class Graphics():


    """La clase Graphics consta de diferentes métodos para la creación de gráficas"""
    def __init__(self):
        pass

    def createSentimentsBarPlot(self, df: pd.DataFrame, sentiment_tag:str = 'sentiment') -> None:
        """Este método crea un histograma para visualizar como se distribuye una variable
            : param df: DataFrame de pandas
            : param sentiment_tag: Variable que se desea graficar, por defecto 'sentiment'

            : return: None (unicamente muestra la gráfica creada)"""
        sent_df = pd.DataFrame(df[sentiment_tag].value_counts(sort=False).sort_index())
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_xlabel(sentiment_tag, fontsize=18)
        ax.set_ylabel('Frecuencia', fontsize=18)
        ax.set_title(f'Distribución de la variable {sentiment_tag}', fontsize=25)
        sent_df.plot(kind='bar', ax=ax)
        plt.show()

    def createFrecuencieWordsBar(self, top_words: list) -> None:
        """Este método crea un gráfico de barras para visualizar la frecuencia de la lista dada de palabras
            : param top_words: list con las palabras mas frecuentes

            : return: None (unicamente muestra la gráfica creada)"""

        words = [w[0] for w in top_words]
        frequencies = [w[1] for w in top_words]

        # Creamos la figura y establecer el tamaño
        fig, ax = plt.subplots(figsize=(10, 8))

        # Configuramos las etiquetas de los ejes y el título
        ax.set_xlabel('Frecuencia', fontsize=16)
        ax.set_ylabel('Palabras', fontsize=16)
        ax.set_title('Gráfica de frecuencia de palabras', fontsize=20)
        # Configuramos el tamaño de las fuentes de las marcas del eje x
        ax.tick_params(axis='x', labelsize=14)
        # Configuramos el tamaño de las fuentes de las marcas del eje x
        ax.tick_params(axis='y', labelsize=14)

        # Crear el gráfico de barras horizontales
        ax.barh(words, frequencies, color='blue', edgecolor='black')

        # Mostrar el gráfico
        plt.show()

    def createHistogram(self, df: pd.DataFrame, variable: str) -> None:
        """Este método crea un histograma para visualizar la frecuencia de la variable dada
            : param df: pd.DataFrame
            : param variable: variable sobre la que se quiere realizar el histograma

            : return: None (unicamente muestra la gráfica creada)"""
        plt.figure(figsize= (15,10))
        plt.xlabel('nº de palabras',fontsize = 18)
        plt.ylabel('Count',fontsize = 18)
        plt.title('Histograma del tamaño de los documentos del corpus',fontsize = 25)
        plt.hist(df[variable], bins=50)
        plt.show()

    def createWorldCloud(self, text: str) -> None:
        """Este método creara una nube de palabras en base a texto pasado por argument
            : param text: String con las palabras a graficar
            : return: None (unicamente muestra la gráfica creada)"""
        # Creamos un objeto WordCloud y generar la nube de palabras
        wordcloud = WordCloud(width=1000, height=600, max_words=200, background_color="white").generate(text)

        # Visualizamos la nube de palabras
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def tsnePlotSimilarWords(self, labels: list, embedding_clusters: np.array, word_clusters: list, a=0.7) -> None:
        """Este método creara un scatter plot de los clusters de las palabras similares
            : param labels: list.  palabras que se analizan
            : param embedding_clusters: np.array 
            : param word_clusters: list. lista de listas con los diferntes clusters
            : return: None (unicamente muestra la gráfica creada)"""

        plt.figure(figsize=(20, 12))
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:,0]
            y = embeddings[:,1]
            plt.scatter(x, y, c=[color], alpha=a, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                             textcoords='offset points', ha='right', va='bottom', size=12)
        plt.legend(loc=4)
        plt.grid(True)
        plt.title('Representación en 2D de los embeddings de algunos clusters de palabras',fontsize = 18)
        # plt.savefig("f/г.png", format='png', dpi=150, bbox_inches='tight')
        plt.show()

