import openai

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = openai.fine_tuning.jobs.create(
    training_file="file-Cw9nux5foKWfsfgS1uuzSs",
    model="gpt-4o-2024-08-06",
    suffix="chicano-translator"
)

print(response.id)  # Save this job ID to check progress

job = openai.fine_tuning.jobs.retrieve("ftjob-eq6Je0wLGPNl0hvEF6OZUBgR")
print(job.fine_tuned_model)