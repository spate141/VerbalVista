from typing import Any
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(text: str, background_color: str = 'black', colormap: str = 'Pastel1') -> Any:
    """
    Generates a word cloud from the given text and returns a matplotlib plot object.

    :param text: The input text from which to generate the word cloud.
    :param background_color: The background color for the word cloud image. Defaults to 'black'.
    :param colormap: The color map to use for the word cloud. Defaults to 'Pastel1'.
    :return: A matplotlib plot object with the generated word cloud.
    """
    wordcloud = WordCloud(
        background_color=background_color, colormap=colormap, width=1024, height=600
    ).generate(text.lower())

    # Display the generated image
    plt.figure(dpi=600)
    plt.imshow(wordcloud)
    plt.axis("off")
    return plt
