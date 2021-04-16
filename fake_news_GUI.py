import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from Fake_News_Detection.fake_news_functions import *
import keras
from keras.models import load_model


tokenizer = load_tokenizer("Fake_News_Detection/fake_news_tokenizer.pickle")
model = load_model("Fake_News_Detection/fake_news_model.h5")



tool = language_tool_python.LanguageTool('en-US')

class MyGridLayout(GridLayout):
    # Initialize inifte keywords
    def __init__(self, **kwargs):
        # Call grid layout constructor
        super(MyGridLayout, self).__init__(**kwargs)

        # Set colums
        self.cols = 1

        # Create a second gridlayout
        self.top_grid = GridLayout()
        self.top_grid.cols = 2


        # Add Twitter text widget
        self.top_grid.add_widget(Label(text="Twitter text: "))
        # Add input box
        self.tweet_text = TextInput(multiline=True)
        self.top_grid.add_widget(self.tweet_text)

        # Add the new top_grid to our app
        self.add_widget(self.top_grid)

        # Create a Submit Button
        self.submit = Button(text="Submit", font_size=32)
        # Bind the Button
        self.submit.bind(on_press=self.press)
        self.add_widget(self.submit)
    def press(self, instance):
        tweet_text = self.tweet_text.text

        # tweet_text_processed = process_single_tweet(tweet_text)
        predict_tweet = predict_real_fake(tweet_text, model, tokenizer)

        self.add_widget(Label(text=f'''{tweet_text}
        {predict_tweet}'''))

        # Clear the input boxes
        self.tweet_text.text = ""


class MyApp(App):
    def build(self):
        return MyGridLayout()

if __name__ == '__main__':
    MyApp().run()
