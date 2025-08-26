# Arguments that aren't required
# --no_cot default value is true
NOT_REQUIRED_ARGS=' --model_name --no_cot'

# Required arguments
REQUIRED_ARGS='--voice_id --input_file --output_dir'

# Run audio generation script with required arguments and any desired unrequired ones
py audio_gen_elevenlabs.py 

#few shot prompts for dialects
