# WIKIPEDIA: TEXT PREPROCESSING AND VISUALIZATION

######################
# Problem Definition
######################

# Perform text preprocessing and visualization on a dataset containing Wikipedia texts.

######################
# Dataset Story
######################

# It contains text taken from Wikipedia data.

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


# Dataset details

df = pd.read_csv("wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape


# STEP 1: PERFORM TEXT PREPROCESSING OPERATIONS
    # Create a function named clean_text for text preprocessing. The function should;
        # Convert uppercase to lowercase letters,
        # Remove punctuation marks,
        # Remove numeric expressions.
    # Apply the function you wrote to all texts in the dataset.

def clean_text(text):
    # Normalizing case folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', ' ')
    text = text.str.replace("\n" , '')
    # Numbers
    text = text.str.replace('\d', '')
    return text

df["text"] = clean_text(df["text"])
df.head()


# Write a function named remove_stopwords that will remove insignificant words while extracting features from the text.
# Apply the function you wrote to all texts in the dataset.

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text
df["text"] = remove_stopwords(df["text"])


pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

# Find words that occur rarely in the text (such as less than 1000, less than 2000). And remove these words from the text.

delete = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

# Tokenize the texts and observe the results.

df["text"].apply(lambda x: TextBlob(x).words)

# Perform lemmatization.

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.head()


# STEP 2: VISUALIZE THE DATA
    # Calculate the frequencies of the terms in the text.
    # Create a Barplot graph of the term frequencies you found in the previous step.

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.head()

tf.columns = ["words", "tf"]
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Visualize the words with WordCloud.

text = " ".join(i for i in df["text"])
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="purple").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# STEP 3: WRITE ALL STEPS AS A SINGLE FUNCTION
    # Perform the text processing operations.
    # Add the visualization operations as an argument to the function.
    # Write a 'docstring' that explains the function.

df = pd.read_csv("wiki_data.csv", index_col=0)
def wiki_preprocess(text, Barplot = False, Wordcloud = False):
    """
    It performs preprocessing operations on text.

    :param text: The variable containing the tests in the DataFrame
    :param Barplot: Barplot visualization
    :param Wordcloud: Wordcloud visualization
    :return: text

    """
    # Normalizing case folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', ' ')
    text = text.str.replace("\n" , '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords
    delete = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in delete))


    if Barplot:
        # Calculate the frequencies of the terms in the text.
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Naming of columns
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        text = " ".join(i for i in text)
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="purple").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])
wiki_preprocess(df["text"], True,True)