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
df.insert(loc=1, column='question_translated', value=None)
df.insert(loc=4, column='passage_translated', value=None)

df = df [["question", "question_translated", "passage", "passage_translated", "answer"]]

few_shot_prompt = """
Translate the following from Standard American English (SAE) to Colloquial Singaporean English (Singlish). Here are a few examples:

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

Now, convert:
"""

def translate_text(text, source_dial="English", target_dial="Colloquial Singaporean English"):
    try:
        full_prompt = few_shot_prompt + f"\nSAE: {text}\nCSE:"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"{full_prompt}"},
            ],
            temperature = 0.3
        )
        time.sleep(0.1)
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error translating text: {e}")
        return text

for i, row in df.iterrows():
    # Translate question if not done
    if pd.isna(row["question_translated"]):
        question_translation = translate_text(row["question"])
        df.at[i, "question_translated"] = question_translation

    # Translate passage if not done
    if pd.isna(row["passage_translated"]):
        passage_translation = translate_text(row["passage"])
        df.at[i, "passage_translated"] = passage_translation

    # Save progress every 10 rows
    if (i + 1) % 10 == 0:
        df.to_excel("SINGAPOREAN_ENGLISH//singlish_boolq.xlsx", index=False)

df.to_excel("SINGAPOREAN_ENGLISH/singlish_boolq.xlsx", index=False)