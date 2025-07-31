import openai
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

job = openai.fine_tuning.jobs.retrieve("ftjob-jM4TQTuyOR20d0oqPmqUpn2x")
print(job.status)
print("Model ID:", job.fine_tuned_model)