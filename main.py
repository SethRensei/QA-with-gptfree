from Voice import Voice

voice = Voice()

if __name__ == "__main__":
    text = voice.listener()
    voice.say(text)