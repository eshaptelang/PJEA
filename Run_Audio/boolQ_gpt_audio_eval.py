import os
import time
import csv
from pathlib import Path
from openai import OpenAI
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIO_DIR = Path("boolq_audios/indian_audio")
OUTPUT_CSV = Path("run_audio_results/GPT_Indian_BoolQ_Results.jsonl")

ds = load_dataset("google/boolq", split="train")

def main():
    results = []

    for idx, example in enumerature(tqdm(ds)):
        filename = f'prompt_{idx}.mp3'
        audio_path = os.path.join(AUDIO_DIR, filename)

        print(f"🔁 Processing {audio_path}...")

        try:
            # Transcribe audio with Whisper
            with open(wav_file, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcription = transcription_response.text
            print(f"📝 Transcription: {transcription}")

            # Send transcription to GPT-4o chat for answer
            chat_response = client.chat.completions.create(
                full_prompt = "Listen to this quewstion and answer True or False. Respond with only one word: True or False" + transcription
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering spoken questions."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            gpt_answer = chat_response.choices[0].message.content
            print(f"🤖 GPT-4o answer: {gpt_answer}")

        except Exception as e:
            print(f"❌ Error processing {wav_file.name}: {e}")
            transcription = "[ERROR]"
            gpt_answer = "[ERROR]"

        results.append({
            "id": sample_id,
            "accent": accent,
            "path": str(wav_file),
            "transcription": transcription,
            "gpt_answer": gpt_answer
        })

        time.sleep(2)  # avoid rate limits

    # Save all results to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n🎉 Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()