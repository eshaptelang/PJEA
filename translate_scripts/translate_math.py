import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
import asyncio
import argparse

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

    if args.target_dial == 'AAVE':
        few_shot_prompt = """
            Translate the following math word problems from Standard American English to African American Vernacular English (AAVE). Keep all numbers, mathematical relationships, and key information exactly the same.

            SAE: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
            AAVE: Janet ducks be laying 16 eggs every day. She eat 3 for breakfast every morning and bake muffins for her friends every day with 4. She sell what's left at the farmers' market daily for $2 per fresh duck egg. How much money she be making every day at the farmers' market?

            SAE: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take to make 3 robes?
            AAVE: A robe take 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber it take to make 3 robes?

            SAE: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
            AAVE: Josh decide to try flipping a house. He buy a house for $80,000 and then put in $50,000 in repairs. This increased the value of the house by 150%. How much profit he make?

            SAE: James writes a 3-page letter to 2 different friends. He uses both sides of the page and writes 450 words per side. How many words has James written?
            AAVE: James write a 3-page letter to 2 different friends. He use both sides of the page and write 450 words per side. How many words James done wrote?

            SAE: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more purple flowers than yellow. How many flowers are there in Mark's garden?
            AAVE: Mark got a garden with flowers. He planted plants of three different colors in it. Ten of them be yellow, and there are 80% more purple flowers than yellow. How many flowers there be in Mark's garden?
            """
    elif args.target_dial == 'Chicano English':
        few_shot_prompt = """
            Translate the following math word problems from Standard American English to Chicano English. Keep all numbers, mathematical relationships, and key information exactly the same.

            SAE: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
            CE: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells what's left at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day at the farmers' market?

            SAE: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take to make 3 robes?
            CE: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber it takes to make 3 robes?

            SAE: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
            CE: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit he make?

            SAE: James writes a 3-page letter to 2 different friends. He uses both sides of the page and writes 450 words per side. How many words has James written?
            CE: James writes a 3-page letter to 2 different friends. He uses both sides of the page and writes 450 words per side. How many words James has wrote?

            SAE: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more purple flowers than yellow. How many flowers are there in Mark's garden?
            CE: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there's 80% more purple flowers than yellow. How many flowers are there in Mark's garden?
            """
    elif args.target_dial == 'Singlish':
        few_shot_prompt = """
            Translate the following math word problems from Standard American English to Colloquial Singaporean English (Singlish). Keep all numbers, mathematical relationships, and key information exactly the same.

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

            SAE: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
            CSE: Janet's ducks lay 16 eggs per day. She eat 3 for breakfast every morning and bake muffins for her friends every day with 4. She sell the remainder at the farmers' market daily for $2 per fresh duck egg. How much money she make every day at the farmers' market?

            SAE: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take to make 3 robes?
            CSE: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber takes to make 3 robes?

            SAE: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
            CSE: Josh decides to try flipping a house. He buy a house for $80,000 and then put in $50,000 in repairs. This increased the value of the house by 150%. How much profit he make?

            SAE: James writes a 3-page letter to 2 different friends. He uses both sides of the page and writes 450 words per side. How many words has James written?
            CSE: James writes a 3-page letter to 2 different friends. He use both sides of the page and write 450 words per side. How many words James has written?

            SAE: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more purple flowers than yellow. How many flowers are there in Mark's garden?
            CSE: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them yellow, and there are 80% more purple flowers than yellow. How many flowers are there in Mark's garden?
            """
    else:
        raise ValueError(f"Unsupported target dialect: {args.target_dial}")

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

    for i, row in df.iterrows():
        if pd.isna(row["question_translated"]):
            print(f"Translating question {i+1}/{len(df)}")
            question_translation = translate_text(row["question"])
            df.at[i, "question_translated"] = question_translation
            if (i + 1) % 10 == 0:
                df.to_excel(args.output_dir, index=False)
                print(f"Progress saved at question {i+1}")
            time.sleep(1)

    df.to_excel(args.output_dir, index=False)
    print(f"Translation complete! Results saved to {args.output_dir}")

if __name__ == '__main__':
    asyncio.run(main())