from elevenlabs.client import ElevenLabs
import os
import pandas as pd

client = ElevenLabs(api_key="enter api key here!!!!")#replace with your api key

df = pd.read_excel("/Users/priynshuaggarwal/Documents/Algoverse/boolq_jamaican_english.xlsx")#location for excel file


for i in range(20):
    question = df.iloc[i, 1] #column 2, but python = 1 (indexing)
    

    audio = client.text_to_speech.convert(
        text=question,
        voice_id="dhwafD61uVd8h85wAZSE",  # Jamaican English voice id
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    

    save_path = f"/Users/priynshuaggarwal/Documents/Algoverse/jamaican_audio/question_{i+1}.mp3" #saving each audio file with a unique name
    with open(save_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    
    print(f"Saved to : {save_path}")