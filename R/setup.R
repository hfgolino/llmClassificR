#' Set up the Python environment for llmClassificR
#'
#' Creates a Python virtual environment using UV and installs all required
#' packages with pinned versions for cross-platform compatibility (Intel Mac,
#' Apple Silicon, Windows, Linux). Only needs to be run once.
#'
#' @param env_name Name of the virtual environment, or a path to the folder
#'   where it should be created (for example \code{"./llm_env"}). Default is
#'   \code{"llm_env"}, created in the user data directory of the package (see
#'   Details).
#' @param python_version Python version to install. Default is \code{"3.11"}.
#' @param modernbert Logical. If \code{TRUE}, installs transformers 4.48.3,
#'   the first version that supports ModernBERT
#'   (\code{"answerdotai/ModernBERT-base"}), still with PyTorch 2.2.2. Use it
#'   for a separate, optional environment, e.g.
#'   \code{setup_llm_env("modernbert_env", modernbert = TRUE)}, and activate it
#'   with \code{activate_llm_env("modernbert_env")} in a fresh R session.
#'   Default \code{FALSE}.
#' @param ask Logical. Ask for confirmation before writing the environment to
#'   disk (default \code{TRUE}). In a non-interactive session, set
#'   \code{ask = FALSE} to confirm explicitly.
#'
#' @details
#' \strong{Where the environment is created.} When \code{env_name} is a name,
#' the environment is created in the user data directory of the package,
#' \code{tools::R_user_dir("llmClassificR", which = "data")}; when it is a
#' path, it is created there. The environment takes several hundred megabytes
#' (mostly PyTorch) and can be removed at any time by deleting its folder.
#' \code{\link{activate_llm_env}} looks for an environment first in the
#' working directory, where versions up to 0.2.3 created it, and then in the
#' user data directory.
#'
#' This function:
#' \enumerate{
#'   \item Creates a virtual environment using \code{uv venv}
#'   \item Installs API packages (groq, voyageai)
#'   \item Installs PyTorch 2.2.2 and transformers (pinned for Intel Mac compatibility)
#'   \item Installs supporting packages (numpy, safetensors, etc.)
#' }
#'
#' Jev (\code{\link{jev_classify}}) and Jina AI (\code{\link{embed_jina}}) are
#' called from pure R and do not need this environment.
#'
#' UV must be installed first. See \url{https://docs.astral.sh/uv/} for
#' installation instructions.
#'
#' @return Invisibly, the path of the environment folder (or \code{NULL} if
#'   nothing was installed). Progress is shown as messages.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' \dontrun{
#' # First time setup (run once)
#' setup_llm_env()
#'
#' # Custom environment name (in the user data directory)
#' setup_llm_env(env_name = "my_env")
#'
#' # Environment in a folder of your choice
#' setup_llm_env(env_name = "~/envs/llm_env")
#'
#' # Optional second environment for ModernBERT
#' setup_llm_env(env_name = "modernbert_env", modernbert = TRUE)
#' }
#'
#' @export
setup_llm_env <- function(env_name = "llm_env", python_version = "3.11",
                          modernbert = FALSE, ask = TRUE) {

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

  # Quote arguments for the shell of this platform (spaces in paths; Windows)
  q <- function(x) shQuote(x, type = if (.Platform$OS.type == "windows") "cmd" else "sh")

  # Where to write the environment (never silently in the working directory)
  env_dir <- .env_dir(env_name, for_setup = TRUE)
  if (ask) {
    if (!interactive()) {
      stop("setup_llm_env() writes a Python environment of several hundred MB to\n  ",
           env_dir, "\nRun it in an interactive session, or set ask = FALSE to confirm.",
           call. = FALSE)
    }
    ok <- utils::askYesNo(paste0("Create the Python environment (several hundred MB) in\n  ",
                                 env_dir, "\n?"))
    if (!isTRUE(ok)) {
      message("Nothing was installed.")
      return(invisible(NULL))
    }
  }
  dir.create(dirname(env_dir), recursive = TRUE, showWarnings = FALSE)

  .say("Creating Python", python_version, "environment:", env_dir, "\n")
  system(paste("uv venv --python", python_version, q(env_dir)))

  # Detect python path (use full path, quoted for spaces)
  py <- .get_python_path(env_dir)

  transformers_pin <- if (modernbert) "transformers==4.48.3" else "transformers>=4.38,<4.45"
  .say("Installing packages (this takes 1-2 minutes)...\n")
  install_cmd <- paste(
    "uv pip install --python", q(py),
    "groq voyageai",
    q("torch==2.2.2"),
    q(transformers_pin),
    q("accelerate>=0.27,<1.0"),
    q("numpy<2"),
    "safetensors sentencepiece protobuf tokenizers"
  )
  result <- system(install_cmd)

  if (result == 0) {
    .say("\nSetup complete! Now run:\n")
    .say("  activate_llm_env(\"", env_name, "\")\n", sep = "")
    .say("  set_groq_key()     # reads GROQ_API_KEY from ~/.Renviron\n")
    .say("  set_voyage_key()   # reads VOYAGE_API_KEY from ~/.Renviron\n")
  } else {
    warning("Installation may have encountered errors. Check the output above.")
  }

  invisible(env_dir)
}


#' Activate the Python environment
#'
#' Points reticulate to the Python interpreter in the UV virtual environment.
#' Must be called at the start of each R session before using any llmClassificR
#' functions that require Python.
#'
#' @param env_name Name of the virtual environment, or the path to its folder.
#'   Default is \code{"llm_env"}. A name is looked up first in the working
#'   directory and then in the user data directory of the package (see
#'   \code{\link{setup_llm_env}}).
#' @param preload_torch Logical. Load PyTorch right away (default \code{TRUE}).
#'   See Details.
#'
#' @details
#' This function calls \code{reticulate::use_python()} to lock the R session
#' to the specified Python interpreter. Because reticulate locks to the first
#' Python it loads, you must call this \emph{before} any other reticulate
#' operations, and you must restart R to switch environments.
#'
#' \strong{Avoiding "R Session Aborted" crashes.} PyTorch ships its own
#' native libraries (for example, the OpenMP runtime \code{libomp}). When
#' PyTorch is loaded late in a session, after other native libraries are
#' already in memory, it can crash the whole R session (RStudio shows
#' "R encountered a fatal error"), typically on macOS. To prevent this,
#' \code{activate_llm_env()}:
#' \itemize{
#'   \item sets \code{KMP_DUPLICATE_LIB_OK=TRUE} on macOS (unless you set it),
#'     so a second OpenMP runtime gives a warning instead of an abort, and
#'     \code{TOKENIZERS_PARALLELISM=false};
#'   \item loads PyTorch immediately (\code{preload_torch = TRUE}), so its
#'     libraries are the first ones in memory. This adds a few seconds.
#' }
#' Call \code{activate_llm_env()} at the very start of the session, right
#' after \code{library(llmClassificR)} and before other packages and analyses.
#'
#' @return Invisible \code{NULL}.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#'
#' @examples
#' \dontrun{
#' # At the start of every R session, before other packages
#' library(llmClassificR)
#' activate_llm_env()
#'
#' # Only API functions needed (Groq, Voyage AI): skip loading PyTorch
#' activate_llm_env(preload_torch = FALSE)
#' }
#'
#' @export
activate_llm_env <- function(env_name = "llm_env", preload_torch = TRUE) {

  py <- .get_python_path(env_name)

  if (!file.exists(py)) {
    stop(
      "Python environment '", env_name, "' not found at: ", py, "\n",
      "Run setup_llm_env() first, or give the path to an existing environment\n",
      "(environments are looked up in the working directory and in\n",
      tools::R_user_dir("llmClassificR", which = "data"), ").",
      call. = FALSE
    )
  }

  # Environment variables must be set BEFORE Python and PyTorch start
  Sys.setenv(TOKENIZERS_PARALLELISM = "false")
  if (Sys.info()[["sysname"]] == "Darwin" && !nzchar(Sys.getenv("KMP_DUPLICATE_LIB_OK"))) {
    Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")
  }

  reticulate::use_python(py, required = TRUE)
  .pkg_env$python <- py      # used by embed_local(isolate = TRUE)
  .say("Activated environment:", env_name, "\n")
  .say("Python:", py, "\n")

  if (preload_torch) {
    if (reticulate::py_module_available("torch")) {
      torch <- reticulate::import("torch")
      .say("PyTorch", torch$`__version__`, "loaded\n")
    } else {
      message("PyTorch is not installed in '", env_name,
              "': local models (embed_local(), embed_sentences()) will not work.")
    }
  }

  invisible(NULL)
}


#' Set Groq API key
#'
#' Stores the Groq API key for the current R session and initializes the
#' Groq client. Get a free key at \url{https://console.groq.com}.
#'
#' @param key Your Groq API key as a character string. Defaults to the
#'   \code{GROQ_API_KEY} environment variable. The recommended workflow is to
#'   store your keys in \code{~/.Renviron} (e.g., with
#'   \code{usethis::edit_r_environ()}), one per line
#'   (\code{GROQ_API_KEY=gsk_...}), restart R and call \code{set_groq_key()}
#'   with no arguments, so keys never appear in scripts you share.
#' @param max_retries Number of automatic retries (with exponential backoff)
#'   for rate-limit (429) and server errors. Default is \code{5}.
#' @param timeout Maximum number of seconds to wait for each Groq request.
#'   A request that takes longer fails with a timeout error and is retried
#'   (up to \code{max_retries} times). Default is \code{120}, twice the
#'   60-second default of the Groq Python SDK, so that reasoning models
#'   answering large batches have time to finish. A single request can also
#'   set its own limit, for example \code{llm_generate(prompt, timeout = 30)}.
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' # Recommended: key stored in ~/.Renviron as GROQ_API_KEY=gsk_...
#' set_groq_key()
#'
#' # Or explicitly (avoid this in shared scripts)
#' set_groq_key("gsk_abc123...")
#' }
#'
#' @export
set_groq_key <- function(key = Sys.getenv("GROQ_API_KEY"), max_retries = 5L,
                         timeout = 120) {
  if (is.null(key) || !nzchar(key)) {
    stop("Please provide a valid Groq API key (or set GROQ_API_KEY in ~/.Renviron).\n",
         "Get one free at https://console.groq.com", call. = FALSE)
  }
  if (!is.numeric(timeout) || length(timeout) != 1 || is.na(timeout) || timeout <= 0) {
    stop("'timeout' must be a positive number of seconds.", call. = FALSE)
  }

  .pkg_env$groq_client <- .make_groq_client(key, max_retries = max_retries,
                                            timeout = timeout)
  .pkg_env$groq_key <- key
  .say("Groq API key set successfully.\n")

  invisible(NULL)
}


#' Set Voyage AI API key
#'
#' Stores the Voyage AI API key for the current R session. Get a free key
#' at \url{https://dash.voyageai.com}.
#'
#' @param key Your Voyage AI API key as a character string. Defaults to the
#'   \code{VOYAGE_API_KEY} environment variable (see \code{\link{set_groq_key}}).
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' set_voyage_key()   # reads VOYAGE_API_KEY from ~/.Renviron
#' }
#'
#' @export
set_voyage_key <- function(key = Sys.getenv("VOYAGE_API_KEY")) {
  if (is.null(key) || !nzchar(key)) {
    stop("Please provide a valid Voyage AI API key (or set VOYAGE_API_KEY in ~/.Renviron).\n",
         "Get one free at https://dash.voyageai.com", call. = FALSE)
  }

  voyageai <- reticulate::import("voyageai")
  .pkg_env$voyage_client <- voyageai$Client(api_key = key)
  .pkg_env$voyage_key <- key
  .say("Voyage AI API key set successfully.\n")

  invisible(NULL)
}


#' Set Jina AI API key
#'
#' Stores the Jina AI API key for the current R session. Jina AI provides
#' high-quality multilingual embeddings with generous rate limits. Get a
#' free key at \url{https://jina.ai/embeddings/}.
#'
#' @param key Your Jina AI API key as a character string. Defaults to the
#'   \code{JINA_API_KEY} environment variable (see \code{\link{set_groq_key}}).
#'
#' @return Invisible \code{NULL}.
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' set_jina_key()   # reads JINA_API_KEY from ~/.Renviron
#' }
#'
#' @export
set_jina_key <- function(key = Sys.getenv("JINA_API_KEY")) {
  if (is.null(key) || !nzchar(key)) {
    stop("Please provide a valid Jina AI API key (or set JINA_API_KEY in ~/.Renviron).\n",
         "Get one free at https://jina.ai/embeddings/", call. = FALSE)
  }

  .pkg_env$jina_key <- key
  .say("Jina AI API key set successfully.\n")

  invisible(NULL)
}


#' Verify llmClassificR setup
#'
#' Checks that the Python environment is active and all required packages
#' are available. Optionally tests API connectivity.
#'
#' @param test_apis Logical. If \code{TRUE}, sends a small test request to
#'   Groq, Voyage AI, Jina AI and TypeSafe (Jev) to verify that the keys work.
#'   Services without a key are reported as FAILED. Default is \code{FALSE}.
#' @param groq_model Groq model used for the test request. The default,
#'   \code{"openai/gpt-oss-20b"}, is available on Groq's free tier (October
#'   2026). It is a reasoning model, so the test allows about 2,000 completion
#'   tokens: a reasoning model spends tokens on its reasoning before the
#'   answer, and a budget of only a few tokens ends with an empty answer.
#'
#' @return A list with version information, returned invisibly.
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' activate_llm_env()
#' verify_setup()
#'
#' # Also test API keys (read from ~/.Renviron)
#' set_groq_key()
#' set_voyage_key()
#' verify_setup(test_apis = TRUE)
#' }
#'
#' @export
verify_setup <- function(test_apis = FALSE, groq_model = "openai/gpt-oss-20b") {

  .say("Checking Python packages...\n")
  .say("----------------------------------------------------------------------\n")

  versions <- list()

  # Check each package
  pkgs <- c("torch", "transformers", "numpy", "groq", "voyageai")
  for (pkg in pkgs) {
    tryCatch({
      mod <- reticulate::import(pkg)
      ver <- tryCatch(mod$`__version__`, error = function(e) "OK")
      versions[[pkg]] <- ver
      .say(sprintf("  %-15s %s\n", paste0(pkg, ":"), ver))
    }, error = function(e) {
      versions[[pkg]] <<- "NOT FOUND"
      .say(sprintf("  %-15s NOT FOUND\n", paste0(pkg, ":")))
    })
  }

  if (test_apis) {
    .say("\nTesting API connectivity...\n")
    .say("----------------------------------------------------------------------\n")

    # Test Groq
    tryCatch({
      .test_groq_api(groq_model)
      .say(sprintf("  Groq API:      OK (%s)\n", groq_model))
    }, error = function(e) {
      .say("  Groq API:      FAILED -", conditionMessage(e), "\n")
    })

    # Test Voyage AI
    tryCatch({
      vo <- .get_voyageai()
      vo$embed(list("test"), model = "voyage-4", input_type = "query")
      .say("  Voyage AI API: OK\n")
    }, error = function(e) {
      .say("  Voyage AI API: FAILED -", conditionMessage(e), "\n")
    })

    # Test Jina AI
    tryCatch({
      jina_key <- .get_jina_key()
      .jina_embed_request(list("test"), model = "jina-embeddings-v3",
                          task = "text-matching", api_key = jina_key)
      .say("  Jina AI API:   OK\n")
    }, error = function(e) {
      .say("  Jina AI API:   FAILED -", conditionMessage(e), "\n")
    })

    # Test TypeSafe (Jev)
    tryCatch({
      jev_list_models()
      .say("  TypeSafe (Jev): OK\n")
    }, error = function(e) {
      .say("  TypeSafe (Jev): FAILED -", conditionMessage(e), "\n")
    })
  }

  .say("\n")
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
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' # After setting the API key (read from ~/.Renviron), queries live
#' set_groq_key()
#' list_models()
#'
#' # Works without API key (static fallback)
#' list_models()
#'
#' # Only models that support strict JSON Schema (see classify_schema())
#' grep("gpt-oss|qwen3", list_models()$model, value = TRUE)
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

      .say(sprintf("Available Groq models (%d):\n", nrow(models_df)))
      .say("----------------------------------------------------------------------\n")
      for (i in seq_len(nrow(models_df))) {
        .say(sprintf("  %-45s %s\n",
                    models_df$model[i],
                    ifelse(is.na(models_df$context_window[i]), "",
                           paste0("(", formatC(models_df$context_window[i],
                                               format = "d", big.mark = ","),
                                  " ctx)"))))
      }
      .say("----------------------------------------------------------------------\n")
      return(invisible(models_df))
    }
  }

  # Static fallback
  # Chat models on Groq's free tier (checked against
  # https://console.groq.com/docs/models and /docs/deprecations, October 2026).
  # Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile for free and
  # developer-tier keys on August 16, 2026, so they are not listed. Preview
  # models can be removed at short notice; always prefer the live list.
  models <- data.frame(
    model = c("openai/gpt-oss-20b", "openai/gpt-oss-120b",
              "openai/gpt-oss-safeguard-20b", "qwen/qwen3.8-27b"),
    display_name = c("GPT-OSS 20B (reasoning)", "GPT-OSS 120B (reasoning)",
                     "GPT-OSS Safeguard 20B (reasoning, safety policies, preview)",
                     "Qwen 3.8 27B (reasoning, preview)"),
    context_window = c(131072L, 131072L, 131072L, 131072L),
    owned_by = c("OpenAI", "OpenAI", "OpenAI", "Alibaba Cloud"),
    type = c("chat", "chat", "chat", "chat"),
    stringsAsFactors = FALSE
  )

  if (!is.null(type)) {
    type <- tolower(type)
    models <- models[models$type == type, , drop = FALSE]
  }

  .say("Available Groq models (static list - set API key for live query):\n")
  .say("----------------------------------------------------------------------\n")
  for (i in seq_len(nrow(models))) {
    .say(sprintf("  %-45s %s\n", models$model[i],
                ifelse(is.na(models$context_window[i]), "",
                       paste0("(", formatC(models$context_window[i],
                                           format = "d", big.mark = ","), " ctx)"))))
  }
  .say("----------------------------------------------------------------------\n")
  invisible(models)
}


# Internal helpers

#' Folder of a Python environment
#'
#' A path (absolute, starting with "~", or containing a separator) is used as
#' given. A name is looked up first in the working directory (where versions
#' up to 0.2.3 created the environment) and then in the user data directory
#' of the package, which is also where setup_llm_env() creates new ones.
#' @noRd
.env_dir <- function(env_name = "llm_env", for_setup = FALSE) {
  if (grepl("[/\\\\]", env_name) || startsWith(env_name, "~")) {
    return(path.expand(env_name))
  }
  local <- file.path(getwd(), env_name)
  if (!for_setup && dir.exists(local)) return(local)
  file.path(tools::R_user_dir("llmClassificR", which = "data"), env_name)
}

#' Get the Python path for a given environment
#' @noRd
.get_python_path <- function(env_name = "llm_env") {
  env_dir <- .env_dir(env_name)
  if (.Platform$OS.type == "windows") {
    file.path(env_dir, "Scripts", "python.exe")
  } else {
    file.path(env_dir, "bin", "python")
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

#' Create the Groq client (Python SDK through reticulate)
#'
#' A separate function so that tests can replace it with a mock.
#' @noRd
.make_groq_client <- function(key, max_retries = 5L, timeout = 120) {
  groq <- reticulate::import("groq")
  groq$Groq(api_key = key, max_retries = as.integer(max_retries),
            timeout = as.numeric(timeout))
}

#' Get the Groq client, with a helpful error if not set
#' @noRd
.get_groq_client <- function() {
  if (is.null(.pkg_env$groq_client)) {
    if (nzchar(Sys.getenv("GROQ_API_KEY"))) {
      set_groq_key()
    } else {
      stop("Groq API key not set. Run set_groq_key() first ",
           "(or set GROQ_API_KEY in ~/.Renviron).", call. = FALSE)
    }
  }
  .pkg_env$groq_client
}

#' Minimal Groq request used by verify_setup()
#'
#' gpt-oss and Qwen 3.8 are reasoning models: they spend completion tokens on
#' reasoning before the answer. .groq_chat() adds 2000 tokens and sets
#' reasoning_effort = "low" for these models, so the request below allows
#' 2064 completion tokens. An empty answer is treated as a failure.
#' @noRd
.test_groq_api <- function(model = "openai/gpt-oss-20b") {
  out <- .groq_chat("Say 'hello' and nothing else.", model = model,
                    max_tokens = 64L)
  if (!nzchar(trimws(out))) {
    stop("The model returned an empty answer (", model, ").", call. = FALSE)
  }
  invisible(out)
}

#' Get the Voyage AI client, with a helpful error if not set
#' @noRd
.get_voyageai <- function() {
  if (is.null(.pkg_env$voyage_client)) {
    if (nzchar(Sys.getenv("VOYAGE_API_KEY"))) {
      set_voyage_key()
    } else {
      stop("Voyage AI API key not set. Run set_voyage_key() first ",
           "(or set VOYAGE_API_KEY in ~/.Renviron).", call. = FALSE)
    }
  }
  .pkg_env$voyage_client
}

#' Get the Jina AI API key, with a helpful error if not set
#' @noRd
.get_jina_key <- function() {
  if (is.null(.pkg_env$jina_key)) {
    env_key <- Sys.getenv("JINA_API_KEY")
    if (nzchar(env_key)) return(env_key)
    stop("Jina AI API key not set. Run set_jina_key() first ",
         "(or set JINA_API_KEY in ~/.Renviron).", call. = FALSE)
  }
  .pkg_env$jina_key
}

#' Format texts as a numbered list for batch prompting
#' @noRd
.format_numbered_texts <- function(texts) {
  texts <- gsub("[\r\n]+", " ", trimws(texts))   # one text per line
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
