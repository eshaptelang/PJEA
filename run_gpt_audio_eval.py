import os
import time
import csv
from pathlib import Path
from openai import OpenAI

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIO_DIR = Path("google_audio_hindi")
OUTPUT_CSV = Path("gpt_whisper_responses.csv")

def main():
    results = []

    for mp3_file in AUDIO_DIR.rglob("passage_*.mp3"):
        accent = mp3_file.parent.name
        sample_id = mp3_file.stem

        print(f"🔁 Processing {wav_file.name} ({accent})...")

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
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering spoken questions."},
                    {"role": "user", "content": transcription}
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
