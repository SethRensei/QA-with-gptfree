import pyttsx3 as p
import speech_recognition as sr

class Voice():

    def say(self, text):
        engine = p.init()
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate )
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[3].id)
        engine.say(text)
        engine.runAndWait()
        
    
    def listener(self):
        
        r = sr.Recognizer()

        with sr.Microphone() as source:
            
            print("Entrain d'écouter...")
            r.pause_threshold = 1
            audio = r.listen(source)
    
        try:
            query = r.recognize_google(audio, language ='fr')
    
        except Exception as e:
            print(e)
            return "None"
        
        return query