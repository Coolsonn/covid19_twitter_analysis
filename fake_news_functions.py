import re
import numpy as np
import pickle
from nltk.corpus import stopwords
import string
import keras
import language_tool_python
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model4 = load_model("fake_news_model.h5")
tool = language_tool_python.LanguageTool('en-US')

def find_mentions(tweet):
    mentions = re.findall(r'@\w+', tweet)
    return mentions

def del_url(text):
    x = re.sub(r'http\S+', "",text)
    return x

def find_hashtags(tweet):
    hashtags = re.findall(r'#\w+', tweet)
    return hashtags

def remove_mentions(text):
    x = re.sub(r'@\w+',' ',text)
    return x

def correct_grammar(text):
    text = tool.correct(text)
    return text


punctuation = string.punctuation

def remove_punctuations(text):
    punctuation = string.punctuation
    text = "".join([punc for punc in text if not punc in punctuation])
    return text

stop_words = stopwords.words("English")

def remove_stopwords(text):
    stop_words = stopwords.words("English")
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])
    return text

def load_tokenizer(file_name):
    with open(file_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tokenizer("fake_news_tokenizer.pickle")

def process_single_tweet(text):
    to_process = text

    to_process = del_url(to_process) #remove any URLs starting with HTTPS
    to_process = remove_mentions(to_process) #remove any @mentions

    to_process = correct_grammar(to_process) #correct grammar mistakes
    to_process = remove_punctuations(to_process) #remove punctuation

    to_process = to_process.lower() #lower text
    to_process = remove_stopwords(to_process) #remove all stopwords

    to_process = tokenizer.texts_to_sequences([to_process])[0] #tokenize the text with the tokenizer trained on our data
    to_process = pad_sequences([to_process], maxlen=50, padding='post')

    return to_process


def predict_real_fake(text): #function for making predictions
    to_predict = process_single_tweet(text)

    prediction = float(model4.predict(to_predict)[0])

    if prediction > 0.5:
        return f"the model is {round(prediction*100, 2)}% sure that this tweet represents Real News about COVID-19"
    else:
        return f"the model is {round((1-prediction)*100,2)}% sure that this tweet represents Fake News about COVID-19"
