# Arguments that aren't required
# no_passage is set as true as default, model is gpt-4o, --temp is 0.3
NOT_REQUIRED_COMMON_ARGS=' --model_name --temp'
NOT_REQUIRED_PQA_ARGS='--no_passage'
#--no_premise default is false (i.e there is a premise)
NOT_REQUIRED_MC_ARGS='--no_premise'

# Required arguments
REQUIRED_COMMON_ARGS='--dataset --target_dial --output_dir'
REQUIRED_MC_ARGS='--n_choices'

# Run audio generation script with required arguments and any desired unrequired ones
py translate_pqa.py 
py translate_mc.py