docker run --rm --pull=always \
  -v /mnt/c/Users/JonLi/Algoverse/PJEA/run_audio_results/algorithim/SAE:/SAE \
  ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /SAE/results_humaneval_to_eval.jsonl \
  --i-just-wanna-run True


