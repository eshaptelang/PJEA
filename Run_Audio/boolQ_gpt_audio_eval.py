import os
import time
import csv
from pathlib import Path
from openai import OpenAI
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dontenv('../.env')
# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIO_DIR = Path("../boolq_audios/sae_audio")
OUTPUT_FILE = Path("../run_audio_results/GPT_SAE_BoolQ_Results.csv")

ds = load_dataset("google/boolq", split="train")
# Added to select specific number of questions
subset = ds.select(range(30))

def main():
    results = []

    for idx, example in enumerate(subset):

        filename = f'prompt_{idx + 1}.mp3'
        audio_path = os.path.join(AUDIO_DIR, filename)

        print(f"🔁 Processing {audio_path}...")

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio file for {filename}")
            continue

        ground_truth = "true" if example["answer"] else "false"
        try:
            # Transcribe audio with Whisper
            with open(audio_path, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcription = transcription_response.text.strip()
            print(f"📝 Transcription: {transcription}")

            full_prompt = "Listen to this question and answer with only one word: True or False" + transcription
            # Send transcription to GPT-4o chat for answer
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering spoken questions."},
                    {"role": "user", "content": full_prompt}
                ]
            )

            gpt_answer = chat_response.choices[0].message.content.strip()
            normalized = gpt_answer.lower()
            print(f"🤖 GPT-4o answer: {gpt_answer}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            transcription = "[ERROR]"
            gpt_answer = "[ERROR]"

        if "true" in normalized:
            pred = True
        elif "false" in normalized:
            pred = False
        else:
            pred = "UNKNOWN"

        results.append({
            "id": idx,
            "transcription": transcription,
            "gpt_answer": gpt_answer,
            "gpt_normalized_answer": pred,
            "ground_truth": ground_truth
        })

        time.sleep(2)  # avoid rate limits

    # Save all results to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n🎉 Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()