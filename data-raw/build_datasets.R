# ==============================================================================
# Script to build package datasets from CSV files
# Run this from the llmClassificR package root directory
# ==============================================================================

# Place your CSV files in data-raw/ before running this script

# --- Extraversion (first-person; openai/gpt-oss-20b) ---
extraversion <- read.csv("data-raw/extraversion_openai.csv", stringsAsFactors = FALSE,
                         encoding = "UTF-8")
usethis::use_data(extraversion, overwrite = TRUE)
cat("Saved extraversion:", nrow(extraversion), "rows\n")

# --- Extraversion (third-person; openai/gpt-oss-120b) ---
extraversion_3p <- read.csv("data-raw/extraversion_openai120b.csv", stringsAsFactors = FALSE,
                            encoding = "UTF-8")
usethis::use_data(extraversion_3p, overwrite = TRUE)
cat("Saved extraversion_3p:", nrow(extraversion_3p), "rows\n")

# --- Extraversion (llama-3.1-8b-instant; course file extraversion.csv) ---
extraversion_llama <- read.csv("data-raw/extraversion_llama.csv", stringsAsFactors = FALSE,
                               encoding = "UTF-8")
usethis::use_data(extraversion_llama, overwrite = TRUE)
cat("Saved extraversion_llama:", nrow(extraversion_llama), "rows\n")

# --- Emotion-laden texts (course file neuroticism.csv) ---
neuroticism <- read.csv("data-raw/neuroticism.csv", stringsAsFactors = FALSE, encoding = "UTF-8")
usethis::use_data(neuroticism, overwrite = TRUE)
cat("Saved neuroticism:", nrow(neuroticism), "rows\n")

# --- Big Five personality descriptions ---
# NOTE: the CSV starts with a byte-order mark; without fileEncoding = "UTF-8-BOM"
# the first column is read as "X...text".
personality_descriptions <- read.csv("data-raw/personality_descriptions.csv",
                                     stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
usethis::use_data(personality_descriptions, overwrite = TRUE)
cat("Saved personality_descriptions:", nrow(personality_descriptions), "rows\n")

# --- Label descriptions (Jev-style criteria) for three tasks ---
label_descriptions <- list(
  extraversion = c(
    Friendly  = "Friendly: warm, welcoming and kind toward other people; makes others feel included.",
    Positive  = "Positive: optimistic and upbeat; looks on the bright side and celebrates progress.",
    Assertive = "Assertive: takes charge, speaks up directly, sets clear expectations and leads decisively.",
    Energetic = "Energetic: full of energy and enthusiasm; fast-paced, lively and active."),
  emotions = c(
    Sad      = "Sad: feeling down, hopeless, empty or lonely.",
    Excited  = "Excited: thrilled, eager and full of anticipation.",
    Angry    = "Angry: irritated, furious or resentful about unfair treatment.",
    Friendly = "Friendly: kind, warm and eager to help or connect with others."),
  bigfive = c(
    Openess           = "Openness: curious, imaginative and creative; seeks new ideas and experiences.",
    Conscientiousness = "Conscientiousness: organized, disciplined, goal-directed and detail-oriented.",
    Extraversion      = "Extraversion: sociable, outgoing and energized by other people.",
    Agreeableness     = "Agreeableness: compassionate, cooperative, kind and supportive.",
    Neuroticism       = "Neuroticism: anxious, worried and sensitive; experiences emotions intensely.")
)
usethis::use_data(label_descriptions, overwrite = TRUE)

# --- Pre-computed MiniLM sentence embeddings (runs examples offline) ---
# Needs the Python environment (activate_llm_env()); takes 1-3 minutes on a CPU.
# Values are rounded to 4 decimals to keep the package small.
unit_round <- function(E) {
  E <- E / sqrt(rowSums(E^2))
  rownames(E) <- NULL
  round(E, 4)
}
texts <- list(
  extraversion = extraversion$text,
  extraversion_3p = extraversion_3p$text,
  extraversion_llama = extraversion_llama$text,
  neuroticism = neuroticism$text,
  personality_descriptions = trimws(personality_descriptions$text)
)
minilm_embeddings <- list(
  model = "sentence-transformers/all-MiniLM-L6-v2",
  texts = lapply(texts, function(x) unit_round(embed_sentences(x))),
  labels = lapply(label_descriptions, function(d) {
    E <- unit_round(embed_sentences(unname(d)))
    rownames(E) <- names(d)
    E
  })
)
usethis::use_data(minilm_embeddings, overwrite = TRUE, compress = "xz")
cat("Saved minilm_embeddings\n")
