#!/bin/bash

# Job Flags
#SBATCH -p mit_normal

# Load modules if needed
source env10/bin/activate

# if no specified experiment name, default is operator first, no reasoning, bullet point format
# or = openrouter, direct = direct provider api call
# streaming = tracking TTFT

# Run your program
python sembench/sembench_movies_q1.py \
  --model "openai/gpt-5.4-nano" \
  --output-dir "21_or_gpt54nano_streaming_none_100" \
  --data "rotten_tomatoes_movie_reviews_random_1000.csv" \
  --ground-truth "rotten_tomatoes_movie_reviews_random_1000_ground_truth.csv" \
  --num-reviews 100 \
  --doc-format "bullet" \
  --reasoning-effort "none" \
  --include-thoughts \
  --disable-cache \
  --streaming \
  --batch-sizes 1 2 5 10 20 25 50 100

python cuad/cuad_adapted.py \
  --model "openai/gpt-5.4-nano" \
  --output-dir "11_or_oc_gpt54nano_streaming_none_100" \
  --num-contracts 100 \
  --split "train" \
  --seed 42 \
  --prompt-order "operator_first" \
  --doc-format "bullet" \
  --reasoning-effort "none" \
  --include-thoughts \
  --disable-cache \
  --streaming \
  --fields-per-call 41 \
  --batch-sizes 1 2 5 10 15 20 30

python tweeteval/tweeteval_irony.py \
  --model "google/gemini-3-flash-preview" \
  --output-dir "2_or_irony_gem3flash_streaming_none_100" \
  --data irony_random_100.csv \
  --num-tweets 100 \
  --reasoning-effort "none" \
  --doc-format "bullet" \
  --include-thoughts \
  --streaming \
  --disable-cache \
  --batch-sizes 1 2 5 10 20 25 50 100

python tweeteval/tweeteval_emotion.py \
  --model "google/gemini-3-flash-preview" \
  --output-dir "4_or_emotion_gem3flash_streaming_none_100" \
  --data emotion_random_100.csv \
  --num-tweets 100 \
  --reasoning-effort "none" \
  --doc-format "bullet" \
  --include-thoughts \
  --streaming \
  --disable-cache \
  --batch-sizes 1 2 5 10 20 25 50 100


