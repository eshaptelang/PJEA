from google.cloud import texttospeech
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_excel("./INDIAN_ENGLISH/boolq_indian_english.xlsx")

def synthesize_text(text):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code='en-IN',
        name='en-IN-Chirp3-HD-Zubenelgenubi',
    )
     
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config,
    )

    save_path = f"./google_audio_indian/question_{i+1}.mp3" #saving each audio file with a unique name
    with open(save_path, "wb") as out:
        out.write(response.audio_content)
    
    print(f"Saved to : {save_path}")

for i in range (5):
    text=df.iloc[i,1]
    synthesize_text(text)