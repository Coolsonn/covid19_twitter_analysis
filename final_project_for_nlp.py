import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty

from fake_news_functions import *
import keras
from keras.models import load_model
import regex as re
import pickle
from nltk.corpus import stopwords
import string
from keras.preprocessing.sequence import pad_sequences
import language_tool_python
from fake_news_functions import del_url, remove_mentions, correct_grammar, remove_punctuations, remove_stopwords
tokenizer = load_tokenizer("fake_news_tokenizer.pickle")
from keras.preprocessing.text import Tokenizer

tool = language_tool_python.LanguageTool('en-US')

class MyLayout(Widget):
    tweet_text = ObjectProperty(None)

    def press(self):
        tweet_text = self.tweet_input.text

        model4 = load_model("fake_news_model.h5")
        predict_tweet = predict_real_fake(tweet_text)
        self.ids.tweet_predicted_label.text = f'''
            tweets text: "{tweet_text}"
        {predict_tweet}'''


        # Clear the input boxes
        self.tweet_input.text = ""


class MyApp(App):
    def build(self):
        return MyLayout()

if __name__ == '__main__':
    MyApp().run()
