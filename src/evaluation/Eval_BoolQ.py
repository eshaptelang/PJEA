import csv
import pandas as pd

# Dataset to evaluate
df = pd.read_csv('../run_audio_results/GPT_SAE_BoolQ_Results.csv')

accuracy = (df['gpt_normalized_answer'] == df['ground_truth']).mean()

print(f"✅ Accuracy: {accuracy*100:.2f}% ({len(df)} examples)")