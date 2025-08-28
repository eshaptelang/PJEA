import os
import time
import csv
from pathlib import Path
from openai import OpenAI
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
import asyncio

load_dontenv('../.env')
# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train")
# Added to select specific number of questions
subset = ds.select(range(30))

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
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output Directory')
    parser.add_argument('--input_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Input Directory')
    parser.add_argument('--answer_loc',
                        type=str,
                        default='',
                        required=True,
                        help='Location of grount_truth/answer file')
    args = parser.parse_args()
    print(args)

    results = []
    label_data = pd.read_excel(args.answer_loc)

    for idx, example in enumerate(args.input_dir):

        filename = f'prompt_{idx + 1}.mp3'
        audio_path = os.path.join(AUDIO_DIR, filename)

        label = label_data[idx]['label']
        print(label)

        print(f"🔁 Processing {audio_path}...")

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio file for {filename}")
            continue

        try:
            # Transcribe audio with Whisper
            with open(audio_path, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcription = transcription_response.text.strip()
            print(f"📝 Transcription: {transcription}")

            full_prompt = ("Listen to the question. You will be told whether or not you are looking for a cause, or an effect. "
                          "You will then be given 2 choices to choose from. Return only 0 if the answer is choice 1 and only 1 if the answer is choice 2. "
                          f"Question: {transcription}")

            # Send transcription to GPT-4o chat for answer
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering spoken questions."},
                    {"role": "user", "content": full_prompt}
                ]
            )

            gpt_answer = chat_response.choices[0].message.content.strip()
            print(f"🤖 GPT-4o answer: {gpt_answer}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            transcription = "[ERROR]"
            gpt_answer = "[ERROR]"

        results.append({
            "id": idx,
            "transcription": transcription,
            "gpt_answer": gpt_answer,
            "gpt_normalized_answer": pred,
            "label": label
        })

        time.sleep(2)  # avoid rate limits

    # Save all results to CSV
    with open(args.output_dir, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n🎉 Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio(main())