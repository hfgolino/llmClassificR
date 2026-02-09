# ==============================================================================
# Script to build package datasets from CSV files
# Run this from the textLLM package root directory
# ==============================================================================

# Place your CSV files in data-raw/ before running this script

# --- Extraversion (first-person) ---
extraversion <- read.csv("data-raw/extraversion_openai.csv", stringsAsFactors = FALSE)
usethis::use_data(extraversion, overwrite = TRUE)
cat("Saved extraversion:", nrow(extraversion), "rows\n")

# --- Extraversion (third-person) ---
extraversion_3p <- read.csv("data-raw/extraversion_openai120b.csv", stringsAsFactors = FALSE)
usethis::use_data(extraversion_3p, overwrite = TRUE)
cat("Saved extraversion_3p:", nrow(extraversion_3p), "rows\n")

# --- Big Five personality descriptions ---
personality_descriptions <- read.csv("data-raw/personality_descriptions.csv", stringsAsFactors = FALSE)
usethis::use_data(personality_descriptions, overwrite = TRUE)
cat("Saved personality_descriptions:", nrow(personality_descriptions), "rows\n")
