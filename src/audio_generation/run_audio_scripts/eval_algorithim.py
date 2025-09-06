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
import base64
import json
from evalplus.data import write_jsonl

load_dotenv('../.env')
# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def decode_rephrased_gen(response, func_name):
        try:
            decoded_response = response.replace('python_function', func_name)
            return decoded_response
        except:
            return None 

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
    parser.add_argument('--data_key',
                        type=str,
                        default='',
                        required=True,
                        help='Path to desired section')
    args = parser.parse_args()
    print(args)

    results = []

    with open(args.answer_loc, 'r') as f:
        dataset=json.load(f)
        for key in args.data_key.split('.'):
            dataset = dataset[key]

    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")

    audio_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mp3')]

    for idx, example in enumerate(audio_files):

        filename = f'prompt_{idx + 1}.mp3'
        audio_path = os.path.join(args.input_dir, filename)

        curr_item = dataset[idx]
        task_id = curr_item['task_idx']
        curr_prompt = curr_item['prompt']
        function_name = curr_item['function_name']
        print(task_id)

        print(f"🔁 Processing {audio_path}...")

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio file for {filename}")
            continue

        # Audio needs to be base 64 encoded before sending to gpt
        with open(audio_path, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that provides solutions to questions. Ensure that the name of your function is python_function"},
                    {"role": "user", "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": "mp3"
                            }
                        }
                    ]}
                ]
            )

            response = chat_response.choices[0].message.content.strip()
            print(f"🤖 GPT-4o answer: {response}")

            # parsing answer
            try:
                parsed_response = response.split("```")[1].strip()
                if parsed_response.startswith('python\n'):
                    parsed_response = parsed_response[7:]
            except:
                parsed_response = response.strip()
            

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            transcription = "[ERROR]"
            gpt_answer = "[ERROR]"

        results.append({
            "task_id": task_id,
            "question": curr_prompt,
            "function_name": function_name,
            "response": response,
            "parsed_answer": parsed_response
        })

        time.sleep(2)  # avoid rate limits

    # saving code
    # changing function names from python_function to correct name for parsed_answer
    results_to_eval = [{'task_id': output['task_id'], 'solution': decode_rephrased_gen(output['parsed_answer'], output['function_name'])} for output in results]
    full_results_to_eval = [{'task_id': output['task_id'], 'solution': decode_rephrased_gen(output['response'], output['function_name'])} for output in results]
    write_jsonl(args.output_dir.replace('.json', f'_humaneval_to_eval.jsonl'), results_to_eval[:164])
    write_jsonl(args.output_dir.replace('.json', f'_humaneval_to_eval_unparsed.jsonl'), full_results_to_eval[:164])
    write_jsonl(args.output_dir.replace('.json', f'_mbpp_to_eval.jsonl'), results_to_eval[164:])
    write_jsonl(args.output_dir.replace('.json', f'_mbpp_to_eval_unparsed.jsonl'), full_results_to_eval[164:])

    print(f"\n🎉 Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())