from elevenlabs.client import ElevenLabs
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

df = pd.read_excel("./SINGAPOREAN_ENGLISH/boolq_singlish.xlsx") # location for excel file

for i in range(5):
    question = df.iloc[i, 3] # column 2, but python = 1 (indexing)

    audio = client.text_to_speech.convert(
        text=question,
        voice_id="7DHxEfwQzXWuiEldRcxx",  # Tristan (Singaporean) Voice ID
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    save_path = f"./SINGAPOREAN_ENGLISH/passage_{i+1}.mp3" # saving each audio file with a unique name
    with open(save_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print(f"Saved to : {save_path}")