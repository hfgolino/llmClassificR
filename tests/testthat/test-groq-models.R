# Groq model IDs, setup checks, timeout and token usage (0.2.3).
# Everything is mocked: no request leaves the computer.

deprecated_groq <- c("llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                     "qwen/qwen3.6-27b", "groq/compound", "groq/compound-mini")

test_that("no exported function defaults to a deprecated Groq model", {
  ns <- asNamespace("llmClassificR")
  for (fn in getNamespaceExports(ns)) {
    f <- get(fn, envir = ns)
    if (!is.function(f)) next
    fm <- formals(f)
    for (arg in intersect(c("model", "models", "groq_model"), names(fm))) {
      default <- tryCatch(eval(fm[[arg]]), error = function(e) NULL)
      if (!is.character(default)) next
      expect_false(any(default %in% deprecated_groq),
                   label = paste0(fn, "(", arg, ") default"))
      expect_false(any(grepl("llama", default, ignore.case = TRUE)),
                   label = paste0(fn, "(", arg, ") default"))
    }
  }
})

test_that("classify_ensemble() calls three free-tier models from two families", {
  local_mock_groq()
  d <- make_texts(2)
  out <- capture.output(ens <- classify_ensemble(d$text, test_classes))
  models <- vapply(mock_log$calls, `[[`, "", "model")
  expect_equal(models, c("openai/gpt-oss-20b", "openai/gpt-oss-120b",
                         "qwen/qwen3.8-27b"))
  expect_true(all(vapply(mock_log$calls, function(cl) identical(cl$reasoning_effort, "low"),
                         logical(1))))
  expect_equal(ens$prediction, d$label)
})

test_that("the static list_models() fallback has no deprecated models", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$groq_client
  pkg$groq_client <- NULL
  withr::defer(pkg$groq_client <- old)
  out <- capture.output(m <- list_models())
  expect_false(any(m$model %in% deprecated_groq))
  expect_false(any(grepl("llama", m$model, ignore.case = TRUE)))
  expect_setequal(m$model, c("openai/gpt-oss-20b", "openai/gpt-oss-120b",
                             "openai/gpt-oss-safeguard-20b", "qwen/qwen3.8-27b"))
  expect_false(anyNA(m$context_window))
  expect_true(any(grepl("static list", out)))
})

test_that("the Groq check of verify_setup() uses gpt-oss-20b with room to reason", {
  local_mock_groq()
  out <- llmClassificR:::.test_groq_api()
  expect_equal(as.character(out), "Some generated prose.")
  call <- mock_log$calls[[1]]
  expect_equal(call$model, "openai/gpt-oss-20b")
  expect_gt(call$max_completion_tokens, 2000)
  expect_equal(call$reasoning_effort, "low")
})

test_that("an empty answer makes the Groq check fail", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$groq_client
  pkg$groq_client <- list(chat = list(completions = list(create = function(...) {
    list(choices = list(list(message = list(content = NULL))))
  })))
  withr::defer(pkg$groq_client <- old)
  expect_error(llmClassificR:::.test_groq_api(), "empty answer")
})

test_that("verify_setup(test_apis = TRUE) reports the Groq model it tested", {
  local_mock_groq()
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- mget(c("voyage_client", "jina_key", "typesafe_key"), envir = pkg,
              ifnotfound = list(NULL))
  for (k in names(old)) assign(k, NULL, envir = pkg)
  withr::defer(for (k in names(old)) assign(k, old[[k]], envir = pkg))
  withr::local_envvar(VOYAGE_API_KEY = "", JINA_API_KEY = "", TYPESAFE_API_KEY = "")
  # No Python needed: every import fails and is reported as NOT FOUND
  local_mocked_bindings(import = function(module, ...) stop("no Python in tests"),
                        .package = "reticulate")
  out <- capture.output(verify_setup(test_apis = TRUE))
  expect_true(any(grepl("Groq API: +OK \\(openai/gpt-oss-20b\\)", out)))
  expect_equal(mock_log$calls[[1]]$model, "openai/gpt-oss-20b")

  out <- capture.output(verify_setup(test_apis = TRUE, groq_model = "openai/gpt-oss-120b"))
  expect_true(any(grepl("OK \\(openai/gpt-oss-120b\\)", out)))
})

test_that("set_groq_key() passes a timeout to the Groq client", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- mget(c("groq_client", "groq_key"), envir = pkg, ifnotfound = list(NULL))
  withr::defer(for (k in names(old)) assign(k, old[[k]], envir = pkg))
  # The Python SDK is replaced by a function that returns its arguments
  local_mocked_bindings(import = function(module, ...) list(Groq = function(...) list(...)),
                        .package = "reticulate")
  out <- capture.output(set_groq_key("test-key"))
  expect_equal(pkg$groq_client$timeout, 120)
  expect_equal(pkg$groq_client$max_retries, 5L)
  expect_equal(pkg$groq_client$api_key, "test-key")
  out <- capture.output(set_groq_key("test-key", timeout = 30, max_retries = 2))
  expect_equal(pkg$groq_client$timeout, 30)
  expect_equal(pkg$groq_client$max_retries, 2L)
  expect_error(set_groq_key("test-key", timeout = 0), "positive number")
  expect_error(set_groq_key("test-key", timeout = "a"), "positive number")
})

test_that("a single request can set its own timeout", {
  local_mock_groq()
  llm_generate("Hello", timeout = 30)
  expect_equal(mock_log$calls[[1]]$extra$timeout, 30)
})

test_that("llm_generate() keeps the token usage and prints only the text", {
  local_mock_groq()
  x <- llm_generate("Hello")
  expect_s3_class(x, "llm_text")
  expect_true(is.character(x))
  expect_equal(attr(x, "usage"), c(prompt_tokens = 100L, completion_tokens = 50L))
  printed <- capture.output(print(x))
  expect_equal(printed, '[1] "Some generated prose."')
  expect_output(cat(x), "^Some generated prose\\.$")
  expect_null(attributes(as.character(x)))
})

test_that("rag_generate() returns the token usage", {
  local_mock_groq()
  local_mocked_bindings(retrieve_similar = function(...) {
    list(texts = c("I love parties.", "I lead the team."), scores = c(0.9, 0.8))
  })
  res <- rag_generate("What themes appear?", doc_embeddings = matrix(0, 2, 2),
                      texts = c("I love parties.", "I lead the team."), top_k = 2)
  expect_named(res, c("answer", "retrieved", "scores", "usage"))
  expect_equal(res$usage, c(prompt_tokens = 100L, completion_tokens = 50L))
  expect_null(attr(res$answer, "usage"))
  expect_equal(mock_log$calls[[1]]$model, "openai/gpt-oss-120b")
})

test_that("list_jina_models() lists the v5 text models", {
  out <- capture.output(m <- list_jina_models())
  v5 <- m[grepl("v5", m$model), ]
  expect_equal(v5$model, c("jina-embeddings-v5-text-small", "jina-embeddings-v5-text-nano"))
  expect_equal(v5$dimensions, c(1024L, 768L))
  expect_true(all(v5$matryoshka))
})

test_that("classify_judge() keeps its default persona when persona = NULL", {
  local_mock_groq()
  d <- make_texts(1)
  res <- classify_judge(d$text, test_classes)
  expect_equal(res$prediction, d$label)
  expect_equal(mock_log$calls[[1]]$system, paste(
    "You are an expert personality psychologist acting as a text classifier.",
    "Your job is to carefully read personality descriptions and classify them into categories.",
    "You must provide reasoning for your classification, citing specific evidence from the text.",
    sep = "\n"))
  res <- classify_judge(d$text, test_classes, persona = "You are an emotion researcher.")
  expect_equal(mock_log$calls[[2]]$system, "You are an emotion researcher.")
})
