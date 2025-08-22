import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# COPA
ds = load_dataset("super_glue", "copa", split="train")
df = ds.to_pandas()

def translate_text(text, source_dial="English", target_dial="African American Vernacular English"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Translate from Standard American English to {target_dial}"},
            {"role": "user", "content": f'SAE: "{text}"\nAAVE:'}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# 29 Examples
subset_df = df.head(29).copy()
subset_df["premise_aave"] = subset_df["premise"].apply(lambda x: translate_text(x))

subset_df["choice1_aave"] = subset_df["choice1"].apply(lambda x: translate_text(x))
subset_df["choice2_aave"] = subset_df["choice2"].apply(lambda x: translate_text(x))
subset_df["label"] = subset_df["label"]

subset_df.to_csv("AAVE_copa_first29.csv", index=False)