import re
import emoji
import contractions as con
import string
import en_core_web_lg
# pip install spacy
# python -m spacy download en_core_web_lg
from autocorrect import Speller
import pandas as pd

nlp=en_core_web_lg.load()
speller=Speller(lang='en')
stop_words=nlp.Defaults.stop_words

def preprocessingText(text):
  text = text.lower()
  # Remove urls
  text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
  # # Remove usernames
  text = re.sub(r'@[^\s]+','', text)
  # # Replace all emojis from the emoji shortcodes
  text = emoji.demojize(text)
  # # Replace chat words and numbers
  text = " ".join([replace_chat_words(word) for word in text.split()])
  # Replace contraction words
  text=con.fix(text)
  # Remove punctuations
  text = "".join([i for i in text if i not in string.punctuation])
  # Replace 3 or more consecutive letters by 1 letter and lemmatizing the words
  text = " ".join([re.sub(r"(.)\1\1+", r"\1", str(token)) if token.pos_ in ["PROPN", 'NOUN'] else token.lemma_ for token in nlp(text)])
  # Replace misspelled words
  text=speller(text)
  # Remove stopwords
  text = " ".join([word for word in text.split() if word not in stop_words])

  text = text.strip()

  return text


from num2words import num2words
slangDf = pd.read_csv("slang.csv")
slangDf=slangDf[['acronym','expansion']]
slangDf.head()

def replace_chat_words(text):
    normal_word=slangDf[slangDf['acronym'].isin([text])]['expansion'].values
    if len(normal_word)>=1:
        if text=='lol':
            return normal_word[1]
        else:
            return normal_word[0]
    elif text.isnumeric():
        return num2words(text)
    else:
        return text
    

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, class_names=['Positive', 'Negative']):
    """
    Plot a confusion matrix using seaborn heatmap visualization.
    
    Parameters:
    y_test: array-like, True labels.
    y_pred: array-like, Predicted labels.
    class_names: list of strings, Class names for the confusion matrix.
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Convert to percentage
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', cbar=True, 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
