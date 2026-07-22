# Low-Latency Semantic Processing with Optimal Prompt-Level Batching

Repo for measuring and modeling how packing multiple documents into a single LLM call (prompt-level batching) affects latency and accuracy for semantic filter and map operators.


## Repo structure

```
cuad/           CUAD contract field-extraction (sem_map)
litsearch/      LitSearch paper filtering (sem_filter)
sembench/       SemBench movie-review sentiment filtering (sem_filter)
tweeteval/      TweetEval irony and emotion (sem_filter / sem_map)
utils/          Shared BatchedExecutor base class + provider clients
                (OpenAI, Gemini, OpenRouter, vLLM)
experiment.sh   Example sweep invoking each task across batch sizes
```

Each task script sweeps a list of batch sizes, issuing `⌈num_docs / batch_size⌉`
calls per size, and logs per-call and per-run latency/accuracy to CSV.

## Setup

```bash
pip install -r requirements.txt
```

Set the relevant API key(s) as environment variables (e.g. `OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`) depending on which provider/model you pass
via `--model`.

## Running

Example calls are written in `experiment.sh`.

## Data

Datasets are not bundled; place them under `data/<name>/` as below before running.

| Dataset | Task | Source | Expected path |
|---|---|---|---|
| SemBench (movie reviews) | sem_filter | Lao et al., [SemBench: A Benchmark for Semantic Query Processing Engines](https://arxiv.org/abs/2511.01716), arXiv:2511.01716 | `data/sembench_movies/*.csv` |
| TweetEval (irony, emotion) | sem_filter / sem_map | Barbieri et al., [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421), Findings of EMNLP 2020 ([cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)) | `data/tweeteval/*.csv` |
| LitSearch (paper abstracts) | sem_filter | Ajith et al., [LitSearch: A Retrieval Benchmark for Scientific Literature Search](https://arxiv.org/abs/2407.18940) ([princeton-nlp/LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch)) | fetched via `litsearch/load_litsearch_data.py` (uses HF `datasets`) |
| CUAD (legal contracts) | sem_map | Hendrycks et al., [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268), NeurIPS 2021 Datasets and Benchmarks | `data/cuad-data/{train_separate_questions,test}.json` |
