import pandas as pd
from datasets import load_dataset
import openai
import time
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train")
df = ds.to_pandas()

few_shot_prompt = """
Translate the following from Standard American English to Chicano English

SAE: You guys
CE: y'all

SAE: caught
CE: catched

SAE: He went
CE: he had went

SAE: They're all in there, aren't they
CE: They're all in there, ain't they

SAE: He doesn't like me
CE: He don't like me
"""

def translate_text(text, source_dial="English", target_dial="Chicano English"):
    response = openai.chat.completions.create(
        model="gpt-4o",
        full_prompt = few_shot_prompt + f"\nSAE: \"{text}\"\nCE:"
        messages=[
            {"role": "user", "content": f"Translate from {source_dial} to {target_dial}."},
        ],
        temperature = 0.3
    )
    return response.choices[0].message.content.strip()

df["question"] = df["question"].apply(lambda x: translate_text(x))
df["passage"] = df["passage"].apply(lambda x: translate_text(x))
df.to_csv("Chicano_boolq.csv", index=False)
