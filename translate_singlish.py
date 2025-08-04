import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train")
df = ds.to_pandas()

few_shot_prompt = """
Translate the following from Standard American English (SAE) to Colloquial Singaporean English (Singlish):

SAE: Can you answer the question or not?
CSE: Answer the question, can or not?

SAE: Khatib is very near my place.
CSE: Khatib very near my place.

SAE: Have they sold it already?
CSE: They sold already ah?

SAE: I have met customers like that before.
CSE: I ever met some customer like that.

SAE: John was scolded by his boss.
CSE: John give his boss scold.

Now convert:
"""

def translate_text(text):
    try:
        full_prompt = few_shot_prompt + f"\nSAE: \"{text}\"\nCSE:"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        time.sleep(0.1)
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error translating text: {e}")
        return text

datasets_dir = "./datasets"
os.makedirs(datasets_dir, exist_ok=True)

df_sample = df.head(50).copy()

df_sample["question_singlish"] = df_sample["question"].apply(translate_text)
df_sample["passage_singlish"] = df_sample["passage"].apply(translate_text)

output_path = os.path.join(datasets_dir, "singlish_boolq50.json")
df_sample.to_json(output_path, orient="records", indent=2)
print(f"Translation complete! Saved to {output_path}")