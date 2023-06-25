from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(text, background_color='black', colormap='Pastel1'):
    # Create WordCloud object
    wordcloud = WordCloud(
        background_color=background_color, colormap=colormap, width=1024, height=600
    ).generate(text.lower())

    # Display the generated image
    plt.figure(dpi=600)
    plt.imshow(wordcloud)
    plt.axis("off")
    return plt
