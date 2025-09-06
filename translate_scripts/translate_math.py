import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
import json
from dotenv import load_dotenv
import asyncio
import argparse
from dialect_prompts import get_prompt

load_dotenv()

print("Loaded key:", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='gpt-4o',
                        required=False,
                        help='Model name')
    parser.add_argument('--dataset',
                        type=str,
                        default='openai/gsm8k',
                        required=False,
                        help='Dataset')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output Directory')
    parser.add_argument('--target_dial',
                        type=str,
                        default='',
                        required=True,
                        help='target dialect')
    parser.add_argument('--temp',
                        type=float,
                        default=0.3,
                        required=False,
                        help='model temperature')
    parser.add_argument('--subset_size',
                        type=int,
                        default=100,
                        required=False,
                        help='number of samples to translate')
    args = parser.parse_args()
    print(args)

    few_shot_prompt = get_prompt(args.target_dial)

    ds = load_dataset("openai/gsm8k", "main")
    df = ds['train'].to_pandas()
    df.insert(loc=1, column='question_translated', value=None)

    if args.subset_size < len(df):
        df = df.head(args.subset_size)
    df = df[["question", "question_translated", "answer"]]

    def translate_text(text, target_dial=args.target_dial):
        full_prompt = few_shot_prompt + f"\nSAE: {text}\n{target_dial}:"
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                temperature=args.temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error translating text: {e}")
            return "[ERROR]"

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    results = []

    for i, row in df.iterrows():
        if pd.isna(row["question_translated"]):
            print(f"Translating question {i+1}/{len(df)}")
            question_translation = translate_text(row["question"])
            df.at[i, "question_translated"] = question_translation

            results.append({
                "id": i,
                "question": row["question"],
                "question_translated": question_translation,
                "answer": row["answer"]
            })

            if (i + 1) % 10 == 0:
                with open(args.output_dir, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Progress saved at question {i+1}")

            time.sleep(1)

    with open(args.output_dir, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Translation complete! JSON results saved to {args.output_dir}")

if __name__ == '__main__':
    asyncio.run(main())