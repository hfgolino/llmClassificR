#' Set up the Python environment for textLLM
#'
#' Creates a Python virtual environment using UV and installs all required
#' packages with pinned versions for cross-platform compatibility (Intel Mac,
#' Apple Silicon, Windows, Linux). Only needs to be run once.
#'
#' @param env_name Name of the virtual environment. Default is \code{"llm_env"}.
#' @param python_version Python version to install. Default is \code{"3.11"}.
#'
#' @details
#' This function:
#' \enumerate{
#'   \item Creates a virtual environment using \code{uv venv}
#'   \item Installs API packages (groq, voyageai)
#'   \item Installs PyTorch 2.2.2 and transformers (pinned for Intel Mac compatibility)
#'   \item Installs supporting packages (numpy, safetensors, etc.)
#' }
#'
#' UV must be installed first. See \url{https://docs.astral.sh/uv/} for
#' installation instructions.
#'
#' @return Invisible \code{NULL}. Prints installation progress.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' # First time setup (run once)
#' setup_llm_env()
#'
#' # Custom environment name
#' setup_llm_env(env_name = "my_env")
#' }
#'
#' @export
setup_llm_env <- function(env_name = "llm_env", python_version = "3.11") {

  # Check UV is available
  uv_check <- system("uv --version", intern = TRUE, ignore.stderr = TRUE)
  if (length(uv_check) == 0) {
    stop(
      "UV is not installed or not on PATH.\n",
      "Install UV first:\n",
      "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n",
      "  Windows:     powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n",
      "Then restart RStudio.",
      call. = FALSE
    )
  }

  cat("Creating Python", python_version, "environment:", env_name, "\n")
  system(paste0("uv venv --python ", python_version, " '", env_name, "'"))

  # Detect python path (use full path, quoted for spaces)
  py <- .get_python_path(env_name)

  cat("Installing packages (this takes 1-2 minutes)...\n")
  install_cmd <- paste0(
    "uv pip install --python '", py, "'",
    " groq voyageai",
    " 'torch==2.2.2'",
    " 'transformers>=4.38,<4.45'",
    " 'accelerate>=0.27,<1.0'",
    " 'numpy<2'",
    " safetensors sentencepiece protobuf tokenizers"
  )
  result <- system(install_cmd)

  if (result == 0) {
    cat("\nSetup complete! Now run:\n")
    cat("  activate_llm_env()\n")
    cat("  set_groq_key('your-key-here')\n")
    cat("  set_voyage_key('your-key-here')\n")
  } else {
    warning("Installation may have encountered errors. Check the output above.")
  }

  invisible(NULL)
}


#' Activate the Python environment
#'
#' Points reticulate to the Python interpreter in the UV virtual environment.
#' Must be called at the start of each R session before using any textLLM
#' functions that require Python.
#'
#' @param env_name Name of the virtual environment. Default is \code{"llm_env"}.
#'
#' @details
#' This function calls \code{reticulate::use_python()} to lock the R session
#' to the specified Python interpreter. Because reticulate locks to the first
#' Python it loads, you must call this \emph{before} any other reticulate
#' operations, and you must restart R to switch environments.
#'
#' @return Invisible \code{NULL}.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' # At the start of every R session
#' activate_llm_env()
#' }
#'
#' @export
activate_llm_env <- function(env_name = "llm_env") {

  py <- .get_python_path(env_name)

  if (!file.exists(py)) {
    stop(
      "Python environment '", env_name, "' not found at: ", py, "\n",
      "Make sure you are in the correct working directory (getwd()),\n",
      "and that you have run setup_llm_env() first.",
      call. = FALSE
    )
  }

  reticulate::use_python(py, required = TRUE)
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  cat("Activated environment:", env_name, "\n")
  cat("Python:", py, "\n")

  invisible(NULL)
}


#' Set Groq API key
#'
#' Stores the Groq API key for the current R session and initializes the
#' Groq client. Get a free key at \url{https://console.groq.com}.
#'
#' @param key Your Groq API key as a character string.
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' set_groq_key("gsk_abc123...")
#' }
#'
#' @export
set_groq_key <- function(key) {
  if (missing(key) || !nzchar(key)) {
    stop("Please provide a valid Groq API key.\n",
         "Get one free at https://console.groq.com", call. = FALSE)
  }

  groq <- reticulate::import("groq")
  .pkg_env$groq_client <- groq$Groq(api_key = key)
  .pkg_env$groq_key <- key
  cat("Groq API key set successfully.\n")

  invisible(NULL)
}


#' Set Voyage AI API key
#'
#' Stores the Voyage AI API key for the current R session. Get a free key
#' at \url{https://dash.voyageai.com}.
#'
#' @param key Your Voyage AI API key as a character string.
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' set_voyage_key("pa-abc123...")
#' }
#'
#' @export
set_voyage_key <- function(key) {
  if (missing(key) || !nzchar(key)) {
    stop("Please provide a valid Voyage AI API key.\n",
         "Get one free at https://dash.voyageai.com", call. = FALSE)
  }

  voyageai <- reticulate::import("voyageai")
  .pkg_env$voyage_client <- voyageai$Client(api_key = key)
  .pkg_env$voyage_key <- key
  cat("Voyage AI API key set successfully.\n")

  invisible(NULL)
}


#' Set Jina AI API key
#'
#' Stores the Jina AI API key for the current R session. Jina AI provides
#' high-quality multilingual embeddings with generous rate limits. Get a
#' free key at \url{https://jina.ai/embeddings/}.
#'
#' @param key Your Jina AI API key as a character string.
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' set_jina_key("jina_abc123...")
#' }
#'
#' @export
set_jina_key <- function(key) {
  if (missing(key) || !nzchar(key)) {
    stop("Please provide a valid Jina AI API key.\n",
         "Get one free at https://jina.ai/embeddings/", call. = FALSE)
  }

  .pkg_env$jina_key <- key
  cat("Jina AI API key set successfully.\n")

  invisible(NULL)
}


#' Verify textLLM setup
#'
#' Checks that the Python environment is active and all required packages
#' are available. Optionally tests API connectivity.
#'
#' @param test_apis Logical. If \code{TRUE}, sends a test request to Groq
#'   and Voyage AI to verify API keys work. Default is \code{FALSE}.
#'
#' @return A list with version information, returned invisibly.
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' activate_llm_env()
#' verify_setup()
#'
#' # Also test API keys
#' set_groq_key("your-key")
#' set_voyage_key("your-key")
#' verify_setup(test_apis = TRUE)
#' }
#'
#' @export
verify_setup <- function(test_apis = FALSE) {

  cat("Checking Python packages...\n")
  cat("----------------------------------------------------------------------\n")

  versions <- list()

  # Check each package
  pkgs <- c("torch", "transformers", "numpy", "groq", "voyageai")
  for (pkg in pkgs) {
    tryCatch({
      mod <- reticulate::import(pkg)
      ver <- tryCatch(mod$`__version__`, error = function(e) "OK")
      versions[[pkg]] <- ver
      cat(sprintf("  %-15s %s\n", paste0(pkg, ":"), ver))
    }, error = function(e) {
      versions[[pkg]] <<- "NOT FOUND"
      cat(sprintf("  %-15s NOT FOUND\n", paste0(pkg, ":")))
    })
  }

  if (test_apis) {
    cat("\nTesting API connectivity...\n")
    cat("----------------------------------------------------------------------\n")

    # Test Groq
    tryCatch({
      client <- .get_groq_client()
      resp <- client$chat$completions$create(
        model = "llama-3.1-8b-instant",
        messages = list(list("role" = "user", "content" = "Say 'hello' and nothing else.")),
        temperature = 0.0,
        max_tokens = 5L
      )
      cat("  Groq API:      OK\n")
    }, error = function(e) {
      cat("  Groq API:      FAILED -", conditionMessage(e), "\n")
    })

    # Test Voyage AI
    tryCatch({
      vo <- .get_voyageai()
      vo$embed(list("test"), model = "voyage-3.5", input_type = "query")
      cat("  Voyage AI API: OK\n")
    }, error = function(e) {
      cat("  Voyage AI API: FAILED -", conditionMessage(e), "\n")
    })

    # Test Jina AI
    tryCatch({
      jina_key <- .get_jina_key()
      .jina_embed_request(list("test"), model = "jina-embeddings-v3",
                          task = "text-matching", api_key = jina_key)
      cat("  Jina AI API:   OK\n")
    }, error = function(e) {
      cat("  Jina AI API:   FAILED -", conditionMessage(e), "\n")
    })
  }

  cat("\n")
  invisible(versions)
}


#' List available Groq models
#'
#' Queries the Groq API to retrieve currently available models. If the Groq
#' API key has not been set (via \code{\link{set_groq_key}}), falls back to
#' a curated static list.
#'
#' @param type Filter by model type: \code{"chat"}, \code{"embedding"},
#'   or \code{NULL} for all. Default is \code{NULL}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{model}{The model identifier (use this in classification functions)}
#'   \item{display_name}{A human-readable model name}
#'   \item{context_window}{Maximum context length in tokens}
#'   \item{owned_by}{Model provider/owner}
#' }
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' # After setting API key, queries live
#' set_groq_key("your-key")
#' list_models()
#'
#' # Works without API key (static fallback)
#' list_models()
#' }
#'
#' @export
list_models <- function(type = NULL) {

  # Try live query if API key is set
  if (!is.null(.pkg_env$groq_client)) {
    models_df <- tryCatch({
      .fetch_groq_models_live()
    }, error = function(e) {
      message("Could not fetch live model list: ", conditionMessage(e))
      message("Falling back to static list.\n")
      NULL
    })

    if (!is.null(models_df)) {
      # Apply type filter
      if (!is.null(type)) {
        type <- tolower(type)
        if (!type %in% c("chat", "embedding")) {
          stop("'type' must be 'chat', 'embedding', or NULL.", call. = FALSE)
        }
        models_df <- models_df[models_df$type == type, , drop = FALSE]
      }

      cat(sprintf("Available Groq models (%d):\n", nrow(models_df)))
      cat("----------------------------------------------------------------------\n")
      for (i in seq_len(nrow(models_df))) {
        cat(sprintf("  %-45s %s\n",
                    models_df$model[i],
                    ifelse(is.na(models_df$context_window[i]), "",
                           paste0("(", formatC(models_df$context_window[i],
                                               format = "d", big.mark = ","),
                                  " ctx)"))))
      }
      cat("----------------------------------------------------------------------\n")
      return(invisible(models_df))
    }
  }

  # Static fallback
  models <- data.frame(
    model = c("llama-3.3-70b-versatile", "llama-3.1-8b-instant",
              "gemma2-9b-it", "deepseek-r1-distill-llama-70b"),
    display_name = c("Llama 3.3 70B", "Llama 3.1 8B",
                     "Gemma 2 9B", "DeepSeek R1 70B"),
    context_window = c(128000L, 131072L, 8192L, 128000L),
    owned_by = c("Meta", "Meta", "Google", "DeepSeek"),
    type = c("chat", "chat", "chat", "chat"),
    stringsAsFactors = FALSE
  )

  if (!is.null(type)) {
    type <- tolower(type)
    models <- models[models$type == type, , drop = FALSE]
  }

  cat("Available Groq models (static list - set API key for live query):\n")
  cat("----------------------------------------------------------------------\n")
  for (i in seq_len(nrow(models))) {
    cat(sprintf("  %-45s %s\n", models$model[i],
                paste0("(", formatC(models$context_window[i],
                                    format = "d", big.mark = ","), " ctx)")))
  }
  cat("----------------------------------------------------------------------\n")
  invisible(models)
}


# Internal helpers

#' Get the Python path for a given environment
#' @noRd
.get_python_path <- function(env_name = "llm_env") {
  if (.Platform$OS.type == "windows") {
    file.path(getwd(), env_name, "Scripts", "python.exe")
  } else {
    file.path(getwd(), env_name, "bin", "python")
  }
}

#' Fetch Groq models live via Python client
#' @noRd
.fetch_groq_models_live <- function() {
  client <- .get_groq_client()
  response <- client$models$list()

  models_raw <- response$data
  n <- length(models_raw)

  if (n == 0) return(NULL)

  model_ids <- character(n)
  display_names <- character(n)
  context_windows <- integer(n)
  owned_by <- character(n)
  model_type <- character(n)

  for (i in seq_len(n)) {
    m <- models_raw[[i]]
    model_ids[i] <- m$id
    display_names[i] <- tryCatch(m$id, error = function(e) NA_character_)
    context_windows[i] <- tryCatch(as.integer(m$context_window), error = function(e) NA_integer_)
    owned_by[i] <- tryCatch(m$owned_by, error = function(e) NA_character_)
    # Classify type based on model id
    model_type[i] <- if (grepl("embed", m$id, ignore.case = TRUE)) "embedding" else "chat"
  }

  df <- data.frame(
    model = model_ids,
    display_name = display_names,
    context_window = context_windows,
    owned_by = owned_by,
    type = model_type,
    stringsAsFactors = FALSE
  )

  # Sort by type then model name
  df <- df[order(df$type, df$model), ]
  rownames(df) <- NULL
  df
}

#' Get the Groq client, with a helpful error if not set
#' @noRd
.get_groq_client <- function() {
  if (is.null(.pkg_env$groq_client)) {
    stop("Groq API key not set. Run set_groq_key('your-key') first.",
         call. = FALSE)
  }
  .pkg_env$groq_client
}

#' Get the Voyage AI client, with a helpful error if not set
#' @noRd
.get_voyageai <- function() {
  if (is.null(.pkg_env$voyage_client)) {
    stop("Voyage AI API key not set. Run set_voyage_key('your-key') first.",
         call. = FALSE)
  }
  .pkg_env$voyage_client
}

#' Get the Jina AI API key, with a helpful error if not set
#' @noRd
.get_jina_key <- function() {
  if (is.null(.pkg_env$jina_key)) {
    stop("Jina AI API key not set. Run set_jina_key('your-key') first.",
         call. = FALSE)
  }
  .pkg_env$jina_key
}

#' Format texts as a numbered list for batch prompting
#' @noRd
.format_numbered_texts <- function(texts) {
  numbered <- paste(seq_along(texts), ". ", texts, sep = "")
  paste(numbered, collapse = "\n")
}

#' Parse batch JSON results into a vector, in order
#' @noRd
.parse_batch_field <- function(results, n, field) {
  out <- character(n)
  for (i in seq_len(n)) {
    entry <- results[[as.character(i)]]
    if (is.null(entry) || is.null(entry[[field]])) {
      out[i] <- NA_character_
    } else {
      out[i] <- tolower(as.character(entry[[field]]))
    }
  }
  out
}
