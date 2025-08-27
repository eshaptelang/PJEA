from elevenlabs.client import ElevenLabs
import os
import pandas as pd
from dotenv import load_dotenv
import argparse
import asyncio

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

#df = pd.read_excel("./INDIAN_ENGLISH/boolq_indian_english.xlsx")#location for excel file
 
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
                        default='eleven_multilingual_v2',
                        required=False,
                        help='Model name')
    parser.add_argument('--voice_id',
                        type=str,
                        default=None,
                        required=True,
                        help='Voice ID'),
    parser.add_argument('--input_file',
                        type=str,
                        default='',
                        required=True,
                        help='Input File')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output Directory')
    parser.add_argument('--no_cot',
                        type=bool,
                        default=False,
                        required=False,
                        help='Disable COT')
    parser.add_argument('--target_column',
                        type=int,
                        default=0,
                        required=True,
                        help='Targeted column to convert')
    args = parser.parse_args()
    print(args)

    df = pd.read_excel(args.input_file)
    print("API Key loaded:", os.getenv("ELEVENLABS_API_KEY")[:10] + "..." if os.getenv("ELEVENLABS_API_KEY") else "None")

    for i in range(0,1):
        question = df.iloc[i, args.target_column] #column 2, but python = 1 (indexing)
        
        if not args.no_cot:
            question = question + " Let's think step by step."

        audio = client.text_to_speech.convert(
            text=question,
            voice_id=args.voice_id, 
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        

        save_path = f"{args.output_dir}/passage_{i+1}.mp3" #saving each audio file with a unique name
        with open(save_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        print(f"Saved to : {save_path}")


if __name__ == '__main__':
    asyncio.run(main())