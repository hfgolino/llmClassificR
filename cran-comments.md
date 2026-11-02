## Test environments

* Ubuntu 24.04, R 4.3.3: `R CMD check --as-cran`
* win-builder: R-devel (2026-11-01 r90825 ucrt)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Notes for the CRAN team

* Most functions call web APIs (Groq, Voyage AI, Jina AI, and TypeSafe's Jev)
  that require API keys, or local transformer models that run in an optional
  Python environment. The examples of these functions are wrapped in
  `\dontrun{}` because they cannot run without API keys or Python. The tests
  mock every API call and run offline, and the functions that do not need
  APIs or Python have runnable examples that use the pre-computed embeddings
  shipped with the package (`minilm_embeddings`).
* `setup_llm_env()` writes a Python environment only when the user calls it,
  after asking for confirmation in interactive sessions (non-interactive
  sessions must pass `ask = FALSE`). By default it writes to
  `tools::R_user_dir("llmClassificR", which = "data")`, or to a path given by
  the user. No example or test calls it.
* Informational output is written with `message()` and can be silenced with
  `suppressMessages()`.
* The words flagged as possibly misspelled in DESCRIPTION are method names
  (Brier score, Platt scaling), an acronym defined in the text (EGA,
  Exploratory Graph Analysis), and the names of the authors of the cited
  reference (Golino and Epskamp).
* The URL https://jina.ai/embeddings flagged on win-builder (redirect) was
  corrected to https://jina.ai/embeddings/.
