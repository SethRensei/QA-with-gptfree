import pyttsx3 as p

class Voice():

    def say(self, text):
        engine = p.init()
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate )
        voices = engine.getProperty('voices')
        engine.setProperty('voices', voices[1].id)
        engine.say(text)
        engine.runAndWait()