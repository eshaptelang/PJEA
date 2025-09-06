import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from dotenv import dotenv_values
import asyncio
import argparse
from dialect_prompts import get_prompt

load_dotenv()

print("Loaded key:", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train")
df = ds.to_pandas()
df.insert(loc=1, column='question_translated', value=None)
df.insert(loc=4, column='passage_translated', value=None)

df = df [["question", "question_translated", "passage", "passage_translated", "answer"]]

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Will add batch processing later
    '''parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        required=False,
                        help='batch size'),'''
    parser.add_argument('--model_name',
                        type=str,
                        default='gpt-4o',
                        required=False,
                        help='Model name')
    parser.add_argument('--dataset',
                        type=str,
                        default='',
                        required=True,
                        help='Dataset')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output Directory')
    parser.add_argument('--no_passage',
                        type=bool,
                        default=True,
                        required=False,
                        help='Is there a passage/context included')
    parser.add_argument('--target_dial',
                        type=str,
                        default='',
                        required=True,
                        help='target dialect')
    parser.add_argument('--temp',
                        type=int,
                        default=0.3,
                        required=False,
                        help='model temperature')
    args = parser.parse_args()
    print(args)

    few_shot_prompt = get_prompt(args.target_dial)

    ds = load_dataset(args.dataset, split="train[:1]")
    df = ds.to_pandas()

    df.insert(loc=1, column='question_translated', value=None)
    if not args.no_passage:
        df.insert(loc=4, column='passage_translated', value=None)
        # formatting for TTS
        df = df [["question", "question_translated", "passage", "passage_translated", "answer"]]


    def translate_text(text, source_dial="English", target_dial=args.target_dial):
        print('hello')
        full_prompt = few_shot_prompt + f"\nSAE: {text}\n{args.target_dial}:"
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "user", "content": f"{full_prompt}"},
            ],
            temperature = args.temp
        )
        return response.choices[0].message.content.strip()

    for i, row in df.iterrows():
        # Translate question if not done
        if pd.isna(row["question_translated"]):
            question_translation = translate_text(row["question"])
            df.at[i, "question_translated"] = question_translation

        # Translate passage if not done
        if not args.no_passage:
            if pd.isna(row["passage_translated"]):
                passage_translation = translate_text(row["passage"])
                df.at[i, "passage_translated"] = passage_translation

        # Save progress every 10 rows
        if (i + 1) % 10 == 0:
            df.to_excel(args.output_dir, index=False)

    df.to_excel(args.output_dir, index=False)

if __name__ == '__main__':
    asyncio.run(main())