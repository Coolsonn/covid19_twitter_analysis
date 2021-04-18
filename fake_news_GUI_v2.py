import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty

from fake_news_functions import *
import keras
from keras.models import load_model

tokenizer_fake_news = load_tokenizer("fake_news_tokenizer.pickle")
tokenizer_sentiment = load_tokenzier("sentiment_tokenizer.pickle")

model_fake_news = load_model("fake_news_model.h5")
model_sentiment = load_model("sentiment_model.h5")

tool = language_tool_python.LanguageTool('en-US')

class MyLayout(Widget):
    tweet_text = ObjectProperty(None)

    def press(self):
        tweet_text = self.tweet_input.text

        predict_fake = predict_real_fake(tweet_text, model_fake_news, tokenizer_fake_news)
        predict_sentiment = predict_sentiment(tweet_text, model_fake_news, tokenizer_fake_news)
        self.ids.tweet_predicted_label.text = f'''
            tweets text: "{tweet_text}"
        {predict_tweet}
        Sentiment: {predict_sentiment}'''


        # Clear the input boxes
        self.tweet_input.text = ""


class MyApp(App):
    def build(self):
        return MyLayout()

if __name__ == '__main__':
    MyApp().run()
