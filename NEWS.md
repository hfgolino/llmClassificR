# llmClassificR 0.2.3

## Groq model updates

Groq deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free
and developer-tier keys on August 16, 2026, and recommends `openai/gpt-oss-20b`
and `openai/gpt-oss-120b` as their replacements
(https://console.groq.com/docs/deprecations). Model IDs were checked against
https://console.groq.com/docs/models in October 2026.

* `verify_setup(test_apis = TRUE)` tests Groq with `openai/gpt-oss-20b`
  instead of `llama-3.1-8b-instant`, so the check no longer fails on free
  keys. The test request allows 2,064 completion tokens (a reasoning model
  spends tokens on reasoning before the answer; the old budget of 5 tokens
  would end with an empty answer), and an empty answer counts as a failure.
  New argument `groq_model` to test another model.
* `classify_ensemble()` defaults to three free-tier models from two families:
  `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`
  (a preview model).
* Every other Groq function that defaulted to `llama-3.3-70b-versatile` now
  defaults to `openai/gpt-oss-120b`: `classify_direct()`,
  `classify_fewshot()`, `classify_cot()`, `classify_json()`,
  `classify_self_consistency()`, `classify_scores()`, `classify_judge()`,
  `llm_generate()`, `rag_generate()`, `discover_topics()` and
  `classify_with_rag()`. These calls now use a reasoning model, so they get
  `reasoning_effort = "low"` and 2,000 extra completion tokens, as before for
  gpt-oss models. Results will differ from those of 0.2.2; pass
  `model = ` explicitly to reproduce older analyses with a key that still has
  access to the retired models.
* The static fallback of `list_models()` no longer lists the deprecated Llama
  models. It lists `openai/gpt-oss-20b`, `openai/gpt-oss-120b`,
  `openai/gpt-oss-safeguard-20b` (preview) and `qwen/qwen3.8-27b` (preview).

## Other changes

* `set_groq_key()` gains `timeout` (default 120 seconds per request; the Groq
  Python SDK default is 60). A request that takes longer fails and is retried
  up to `max_retries` times. A single call can set its own limit, for example
  `llm_generate(prompt, timeout = 30)`.
* `llm_generate()` keeps the token counts reported by Groq in the `"usage"`
  attribute (`prompt_tokens`, `completion_tokens`) instead of dropping them.
  The result now has class `llm_text`, whose print method shows the text
  only; `cat()`, `paste()` and other string functions work as before, and
  `as.character()` returns a plain string.
* `rag_generate()` returns the token counts in a new `usage` element.
* `embed_jina()` and `list_jina_models()` document the Jina v5 text models,
  `jina-embeddings-v5-text-small` (1024 dimensions, 32,768 tokens) and
  `jina-embeddings-v5-text-nano` (768 dimensions, 8,192 tokens). The default
  stays `jina-embeddings-v3`.
* `classify_judge()`: the default of `persona` is now `NULL`, which uses the
  same personality-psychologist persona as before (the long default string
  made the usage line too wide for the manual). Results are unchanged.
* The author line of the help pages uses `\email{}`, so the HTML manual
  validates (the address was read as an unknown HTML tag).
* README: notes for 0.2.3, citation updated to 0.2.3, and grDevices, callr
  and clue added to the list of required packages.
* The examples of `list_models()` and `embed_jina()` call `set_groq_key()` and
  `set_jina_key()` with no argument, so keys are read from `~/.Renviron`
  instead of being typed in scripts.

# llmClassificR 0.2.2

* `prime_weight_encode()` gains `remove_floor` and returns the floor-free
  network. Every layer network adds $p^0 = 1$ to the pairs of texts it does
  not link, so pairs never linked get the number of networks $L$ (the floor)
  and walktrap sees a constant edge between every pair. The new `network`
  element is the floor-free encoding rescaled to 0-1,
  $R = (X - L) / (\sum_k p_k - L)$, with a zero diagonal.
  `remove_floor = TRUE` runs walktrap on it; the default (`FALSE`) keeps the
  raw encoding, so results from earlier versions are unchanged. The output
  also gains `remove_floor` and `primes`.
* New `plot_prime_weight()`: draws the floor-free prime-weight network with
  EGAnet, colored by the communities. With `true_labels`, the communities are
  renamed after the labels they overlap most, and a second panel shows the
  same network colored by the true labels, with the same layout and the same
  color per label (texts that change color are the disagreements). Works with
  objects from earlier versions too.
* New `match_communities()`: renames communities after the true labels with
  the Hungarian algorithm (one label per community, maximum agreement);
  leftover communities are named "Extra k".
* clue (Hungarian algorithm) and grDevices are now imported.

# llmClassificR 0.2.1

* Fixes "R Session Aborted" crashes in RStudio when `embed_local()` runs late
  in a session (reported on macOS). PyTorch ships its own native libraries
  (e.g., the OpenMP runtime) that can clash with libraries already loaded:
  * `embed_local()` now runs, by default, in a separate background R process
    (`isolate = TRUE`, via callr). A PyTorch crash there cannot abort the
    user's session; it becomes an ordinary R error that shows the background
    process's last messages (e.g., "OMP: Error #15"). `isolate = FALSE` runs
    in the current session as before.
  * `activate_llm_env()` sets `KMP_DUPLICATE_LIB_OK=TRUE` on macOS (unless the
    user set it) and loads PyTorch right away (`preload_torch = TRUE`), so its
    libraries are loaded first. Call it right after `library(llmClassificR)`.
* `embed_local()` mean-pools all layers inside PyTorch and returns one matrix
  per text to R (one conversion instead of one per layer); results are
  identical.
* callr is now imported.

# llmClassificR 0.2.0

Update for the 2026 edition of the course *Text Classification with LLMs in
R*, inspired by Raschka (2026), "Language Models for Text Classification:
From Bag-of-Words to Jev".

## New features

### Jev (TypeSafe AI), pure R

* `set_typesafe_key()` stores the TypeSafe key (default: `TYPESAFE_API_KEY`).
* Question builders `jev_choice()`, `jev_noul()` and `jev_score()`.
* `jev_ask()` (several questions about one text in one call) and
  `jev_ask_many()` (many texts in parallel, with retries on 408/429/5xx/529,
  throttling, and a time/failure/token/cost summary). Failed requests are
  returned as `NULL` instead of stopping the batch.
* High-level wrappers with the same output style as `classify_*()`:
  `jev_classify()` (Choice), `jev_multilabel()` (Noul) and `jev_rate()`
  (Score).
* Parsers `jev_parse_choice()`, `jev_parse_noul()`, `jev_parse_score()`,
  plus `jev_usage()` and `jev_list_models()`.

### Evaluation and calibration

* `evaluate_classification()`: accuracy, macro-F1, Cohen's kappa, NMI and
  the number of invalid answers (never silently dropped).
* `compute_nmi()`: NMI without aricode, for character labels, factors or
  cluster numbers; `NA` handled as singletons (EGA), one category (invalid
  LLM answers) or dropped.
* `calibration_metrics()`, `expected_calibration_error()`, `brier_score()`,
  `reliability_table()`, `plot_reliability()`, `softmax()`,
  `fit_temperature()`, `fit_platt()`, `roc_auc()` and `rlcr_reward()`.

### LLM classification

* `classify_schema()`: strict JSON Schema (constrained decoding) with the
  classes as an `enum`, for `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and
  `qwen/qwen3.8-27b`. `classification_schema()` returns the schema.
* `classify_self_consistency()`: vote probabilities from repeated runs with
  different text orders.
* `llm_generate()` passes extra arguments (`...`) to the Groq API, e.g. the
  `browser_search` tool of `openai/gpt-oss-120b` or `top_p`.
* `run_in_chunks()` and `normalize_labels()` are exported for building your
  own batch classifiers.

### Baselines, probes and a DIY Jev-like classifier

* `bow_features()`, `bow_baseline()`, `top_terms()`, `cv_logistic()`
  (with class probabilities) and `learning_curve()`.
* `linear_probe()` and `probe_layers()` (linear head vs. EGA, layer by
  layer).
* `embed_sentences()` (local MiniLM sentence embeddings), `cosine_scores()`,
  `classify_by_description()` (zero-shot from label descriptions) and
  `train_shared_head()` with a `predict()` method.

### Synthetic data

* `generate_labeled_texts()`: labeled synthetic texts with a strict JSON
  Schema.
* `diversity_report()`: duplicates, lexical diversity, distinct-2,
  within-label overlap and repeated openings.

### Data

* `extraversion_llama` (llama-3.1-8b), `neuroticism` (Sad, Excited, Angry,
  Friendly), `label_descriptions` (Jev-style criteria for three tasks) and
  `minilm_embeddings` (pre-computed embeddings, so the calibration, DIY and
  probe examples run offline).

## Behavior changes

* All `classify_*()` functions now **shuffle** the texts and send them in
  **chunks** (`chunk_size = 40`, `shuffle = TRUE`, `seed = 2026`; the global
  random state is restored). Results are always returned in the original
  order. Use `chunk_size = NULL, shuffle = FALSE` for the 0.1.0 behavior.
* Predictions are mapped to the given `classes` with their **original
  spelling** (0.1.0 returned lowercase labels, which never matched
  capitalized true labels). Answers that are not one of the classes, missing
  answers and failed chunks become `NA` (with a message or warning) instead
  of an error or a new class. If every chunk fails (for example, a wrong API
  key or model name), the function stops with the first error message.
* `classify_scores()` returns one column per class named exactly as in
  `classes` (0.1.0 used whatever keys the model returned, in lowercase).
* `compare_methods()` computes NMI with `compute_nmi()` and also reports
  macro-F1, kappa and invalid answers. Accuracy ignores case and punctuation.
* `set_groq_key()`, `set_voyage_key()` and `set_jina_key()` read the key from
  `GROQ_API_KEY`, `VOYAGE_API_KEY` and `JINA_API_KEY` by default, and the
  functions that need a key pick it up from the environment automatically.
  `set_groq_key()` configures 5 automatic retries (`max_retries`).
* `embed_voyage()`, `retrieve_similar()`, `rag_generate()`,
  `discover_topics()` and `classify_with_rag()` default to `voyage-4`
  (January 2026). Documents and queries must use the same model family;
  re-embed documents created with `voyage-3.5`.

## Improvements and bug fixes

* `compare_methods()` failed with aricode >= 1.1.0 ("NA are not supported"
  for character labels). Fixed.
* A missing answer for one text (e.g., `{"1": ..., "3": ...}`) made
  `classify_*()` stop with "replacement has length zero". Fixed.
* Groq calls use `max_completion_tokens` (`max_tokens` is deprecated), and
  reasoning models (gpt-oss, qwen3) get `reasoning_effort = "low"` plus
  room to think, so their JSON is not cut off.
* `list_models()`: the static fallback lists current production models
  (gemma2-9b-it and deepseek-r1-distill-llama-70b were retired).
* `verify_setup(test_apis = TRUE)` tests voyage-4 and TypeSafe (Jev).
* `embed_voyage()` gains `output_dimension` and batches long inputs;
  `embed_jina()` retries on rate limits; `embed_local()` no longer pads every
  text to `max_length`.
* `ega_classify()`: `n_communities` no longer counts unassigned texts (`NA`)
  as a community.
* `evaluate_layers()`: `explained_var` was always 1 (`prcomp(rank. = 4)`
  does not shorten `sdev`); it now is the share of variance of the first
  four principal components, as documented.
* `discover_topics()` accepts both topic objects and plain arrays of labels,
  and removes duplicates.
* `prime_weight_encode()` gains `true_labels` to report the NMI of every
  walktrap step (diagnostic only; the chosen step stays label-free), and its
  documentation now lists the values it actually returns.
* `ega_classify()` documents that `EGA.fit` chooses the steps by TEFI.
* `setup_llm_env()` quotes arguments correctly on Windows, and gains
  `modernbert = TRUE` for an optional ModernBERT environment
  (transformers 4.48.3 with torch 2.2.2).
* Numbered batch prompts put each text on one line (line breaks inside a
  text no longer break the numbering).
* `%||%` is now defined inside the package (it was used but undefined on
  R < 4.4).
* New offline test suite (testthat) with mocked Groq and TypeSafe APIs.

# llmClassificR 0.1.0

* Initial release.
