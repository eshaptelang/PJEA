import openai
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Upload the training file
with open("chicano_training_data.jsonl", "rb") as file:
    file_response = openai.files.create(
        file=file,
        purpose="fine-tune"
    )

print("Uploaded file ID:", file_response.id)