import pandas as pd
from datasets import load_dataset
import openai
import time
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train[:5]")
df = ds.to_pandas()

def translate_text(text, source_dial="English", target_dial="Chicano English"):
    response = openai.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:algoverse:chicano-translator:BzAehkfn",
        messages=[
            {"role": "system", "content": f"You are a translation assistant. Translate from {source_dial} to {target_dial}."},
            {"role": "user", "content": text}
        ],
        temperature = 0.3
    )
    return response.choices[0].message.content.strip()

df["question"] = df["question"].apply(lambda x: translate_text(x))
df["passage"] = df["passage"].apply(lambda x: translate_text(x))
df.to_csv("Chicano_boolq.csv", index=False)
