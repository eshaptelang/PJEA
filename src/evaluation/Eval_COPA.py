import csv
import pandas as pd
import argparse
import asyncio

# Dataset to evaluate

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file',
                        type=str,
                        default='gpt-4o',
                        required=False,
                        help='results file location')

    df = pd.read_csv(args.results_file)

    accuracy = (df['gpt_answer'] == df['label']).mean()

    print(f"✅ Accuracy: {accuracy*100:.2f}% ({len(df)} examples)")

if __name__ == "__main__":
    asyncio.run(main())
