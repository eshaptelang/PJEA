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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_math_dataset(dataset_name, subset_size=None, random_seed=42):
    """Load and standardize math datasets"""
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        df = ds["train"].to_pandas()
    elif dataset_name == "svamp":
        ds = load_dataset("ChilleD/SVAMP")
        df = ds["train"].to_pandas()
        df = df.rename(columns={'question_concat': 'question', 'Answer': 'answer'})
        df = df[['question', 'answer']]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if subset_size and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=random_seed).reset_index(drop=True)

    return df

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        choices=['gsm8k', 'svamp'],
                        required=True,
                        help='Math dataset to translate')
    parser.add_argument('--dialect',
                        type=str,
                        choices=['AAVE', 'Chicano', 'Chicano English', 'Singlish'],
                        required=True,
                        help='Target dialect')
    parser.add_argument('--subset_size',
                        type=int,
                        required=False,
                        help='number of samples to translate')
    parser.add_argument('--random_seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducible sampling')
    parser.add_argument('--model_name',
                        type=str,
                        default='gpt-4o',
                        required=False,
                        help='Model name')
    parser.add_argument('--temp',
                        type=float,
                        default=0.3,
                        required=False,
                        help='model temperature')

    args = parser.parse_args()

    print("=" * 60)
    print("MATH DATASET TRANSLATION")
    print("=" * 60)
    print(f"Dataset:        {args.dataset}")
    print(f"Target Dialect: {args.dialect}")
    print(f"Model:          {args.model_name}")
    print(f"Temperature:    {args.temp}")
    if args.subset_size:
        print(f"Subset Size:    {args.subset_size}")
        print(f"Random Seed:    {args.random_seed}")
    else:
        print(f"Subset Size:    All questions")
    print("-" * 60)

    df = load_math_dataset(args.dataset, args.subset_size, args.random_seed)
    total_questions = len(df)

    print(f"Loaded {total_questions} questions from {args.dataset}")
    if args.subset_size and args.subset_size < total_questions:
        print(f"Using random sample with seed {args.random_seed}")
    print("=" * 60)

    df.insert(loc=1, column='question_translated', value=None)

    output_dir = f"../../data/translated/{args.dataset}/"
    dialect_name = args.dialect.lower().replace(' ', '_')
    if args.subset_size:
        dialect_filename = f"{dialect_name}_{args.subset_size}.json"
    else:
        dialect_filename = f"{dialect_name}.json"
    output_path = os.path.join(output_dir, dialect_filename)

    os.makedirs(output_dir, exist_ok=True)

    few_shot_prompt = get_prompt(args.dialect)

    def translate_text(text):
        full_prompt = few_shot_prompt + f"\nSAE: {text}\n{args.dialect}:"
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

    results = []

    for i, row in df.iterrows():
        print(f"Translating question {i+1}/{total_questions}")
        question_translation = translate_text(row["question"])

        results.append({
            "id": i,
            "question": row["question"],
            "question_translated": question_translation,
            "answer": row["answer"]
        })

        if (i + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Progress saved: {i+1} questions translated")

        time.sleep(1)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Translation complete! Results saved to {output_path}")
    print(f"Successfully translated {len(results)} questions from {args.dataset} to {args.dialect}")
    if args.subset_size:
        print(f"Random sample used with seed: {args.random_seed}")

if __name__ == '__main__':
    asyncio.run(main())