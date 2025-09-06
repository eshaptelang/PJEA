import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from dotenv import dotenv_values
import asyncio
import argparse
from tqdm import tqdm
from dialect_prompts import get_prompt

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    parser.add_argument('--n_choices',
                        type=int,
                        default=0,
                        required=True,
                        help='Number of choices')
    parser.add_argument('--no_premise',
                        type=bool,
                        default=False,
                        required=False,
                        help='Is there a premise')
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

    # Change :1 into whatever number of data points you want to run, get rid of it for all
    # Leave 'super_glue' for now for testing
    ds = load_dataset('super_glue', args.dataset, split="train[:30]")
    df = ds.to_pandas()

    choice_columns=[]
    # inserting premise_translated column and as many choice{i}_translated columns as needed
    if not args.no_premise:
        df['premise_translated'] = None
    for i in range(args.n_choices):
        df[f"choice{i+1}_translated"] = None
        choice_columns.append(f"choice{i+1}")
    df['question_translated'] = None

    # removing idx column
    del df['idx']

    def translate_text(text, source_dial="English", target_dial=args.target_dial):
        full_prompt = few_shot_prompt + f"\nSAE: {text}\n{args.target_dial}:"
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "user", "content": f"{full_prompt}"},
            ],
            temperature = args.temp
        )
        return response.choices[0].message.content.strip()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Translating rows"):
        # Translate question if not done
        if pd.isna(row["question_translated"]):
            question_translation = translate_text(row["question"])
            df.at[i, "question_translated"] = question_translation

        # Translate premise/passage if exists
        if not args.no_premise:
            if pd.isna(row["premise_translated"]):
                question_translation = translate_text(row["premise"])
                df.at[i, "premise_translated"] = question_translation

        # Translate choices if not done
        for choice_col in choice_columns:
            if pd.isna(row[f"{choice_col}_translated"]):
                passage_translation = translate_text(row[choice_col])
                df.at[i, f"{choice_col}_translated"] = passage_translation


        # Save progress every 10 rows
        if (i + 1) % 10 == 0:
            df.to_excel(args.output_dir, index=False)

    df.to_excel(args.output_dir, index=False)

if __name__ == '__main__':
    asyncio.run(main())