import pyttsx3 as p
import speech_recognition as sr

class Voice():

    def say(self, text):
        """Cette méthode permet de transformer du texte en audio (syntèse vocale)

        Args:
            text (_type_): _description_
        """        
        engine = p.init()
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate )
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[3].id)
        engine.say(text)
        engine.runAndWait()
        
    
    def listener(self):
        """Cette méthode permet d'écouter sur le microphone interne (reconnaissance vocale)

        Returns:
            Any | Literal['None'] : La transciption de l'audio en texte
        """        
        
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.pause_threshold = 1
            audio = r.listen(source)
    
        try:
            query = r.recognize_google(audio, language ='fr')
    
        except Exception as e:
            return "None"
        
        return query