test_that("classify_direct: shuffled chunks, original order, validated labels", {
  local_mock_groq()
  d <- make_texts(5)
  d$text[2] <- paste(d$text[2], "WEIRD")
  d$text[7] <- paste(d$text[7], "MISSING")
  d$text[12] <- paste(d$text[12], "INVALID")
  expect_message(res <- classify_direct(d$text, test_classes, chunk_size = 8), "2 of 20")
  expect_named(res, c("text_id", "text", "prediction"))
  expect_equal(res$text_id, 1:20)
  expect_equal(res$prediction[-c(7, 12)], d$label[-c(7, 12)])
  expect_true(is.na(res$prediction[7]) && is.na(res$prediction[12]))
  expect_length(mock_log$calls, 3)                        # 20 texts / 8 per chunk
  call <- mock_log$calls[[1]]
  expect_false(is.null(call$max_completion_tokens))       # not max_tokens
  expect_equal(call$model, "openai/gpt-oss-120b")         # free-tier default (0.2.3)
  expect_equal(call$reasoning_effort, "low")              # default is a reasoning model
  # shuffled: the first chunk is not simply texts 1 to 8
  first_ids <- as.integer(sub("^Text ([0-9]+) .*", "\\1",
                              regmatches(call$prompt, gregexpr("Text [0-9]+ ", call$prompt))[[1]]))
  expect_false(identical(first_ids, 1:8))
})

test_that("classify_direct with chunk_size = NULL and shuffle = FALSE sends one call", {
  local_mock_groq()
  d <- make_texts(3)
  res <- classify_direct(d$text, test_classes, chunk_size = NULL, shuffle = FALSE)
  expect_equal(res$prediction, d$label)
  expect_length(mock_log$calls, 1)
})

test_that("a chunk with invalid JSON gives NA (with a warning), not an error", {
  local_mock_groq()
  d <- make_texts(3)
  d$text[1] <- paste(d$text[1], "BADJSON")
  expect_warning(res <- suppressMessages(classify_direct(d$text, test_classes, chunk_size = 4,
                                                         shuffle = FALSE)), "Chunk 1 failed")
  expect_true(all(is.na(res$prediction[1:4])))
  expect_equal(res$prediction[5:12], d$label[5:12])
})

test_that("reasoning models get reasoning_effort and extra tokens", {
  local_mock_groq()
  d <- make_texts(2)
  res <- classify_direct(d$text, test_classes, model = "openai/gpt-oss-20b")
  expect_equal(mock_log$calls[[1]]$reasoning_effort, "low")
  expect_gt(mock_log$calls[[1]]$max_completion_tokens, 2000)
  # A model that is not a reasoning model gets neither
  res <- classify_direct(d$text, test_classes, model = "some-chat-model")
  expect_null(mock_log$calls[[2]]$reasoning_effort)
  expect_lt(mock_log$calls[[2]]$max_completion_tokens, 2000)
})

test_that("few-shot, CoT, JSON and judge return tidy data frames", {
  local_mock_groq()
  d <- make_texts(3)
  ex <- data.frame(text = c("Ex Friendly", "Ex Positive"), label = c("Friendly", "Positive"))
  fs <- classify_fewshot(d$text, test_classes, ex)
  expect_equal(fs$prediction, d$label)
  expect_warning(classify_fewshot(d$text, test_classes,
                                  data.frame(text = d$text[1], label = "Friendly")), "also in")
  expect_error(classify_fewshot(d$text, test_classes, data.frame(x = 1)), "text")

  d$text[4] <- paste(d$text[4], "MISSING")
  cot <- suppressMessages(classify_cot(d$text, test_classes))
  expect_named(cot, c("text_id", "text", "prediction", "reasoning"))
  expect_true(is.na(cot$prediction[4]) && is.na(cot$reasoning[4]))
  expect_equal(mock_log$calls[[length(mock_log$calls)]]$n_messages, 2)   # system + user

  js <- suppressMessages(classify_json(d$text, test_classes))
  expect_equal(unique(js$confidence[-4]), "medium")

  jd <- suppressMessages(classify_judge(d$text, test_classes))
  expect_equal(unique(jd$confidence[-4]), "high")               # "High" normalized
  expect_equal(jd$evidence[1], "phrase one; phrase two")
  expect_true(is.na(jd$evidence[4]))
})

test_that("classify_schema aligns answers by id", {
  local_mock_groq()
  d <- make_texts(4)
  res <- classify_schema(d$text, test_classes, chunk_size = 7)
  expect_equal(res$prediction, d$label)                    # mock returns reversed order
  expect_equal(unique(res$confidence), "high")
  expect_equal(mock_log$calls[[1]]$response_format$type, "json_schema")
  res2 <- classify_schema(d$text, test_classes, with_confidence = FALSE)
  expect_named(res2, c("text_id", "text", "prediction"))
})

test_that("classify_scores returns one column per class with original names", {
  local_mock_groq()
  d <- make_texts(3)
  d$text[5] <- paste(d$text[5], "MISSING")
  res <- suppressMessages(classify_scores(d$text, test_classes))
  expect_named(res, c("text_id", "prediction", test_classes))
  expect_equal(res$prediction[-5], d$label[-5])
  expect_true(is.na(res$prediction[5]))
  cm <- calibration_metrics(as.matrix(res[, test_classes]), d$label)
  expect_equal(cm$n_missing, 1)
})

test_that("ensemble and self-consistency combine runs", {
  local_mock_groq()
  d <- make_texts(3)
  out <- capture.output(ens <- classify_ensemble(d$text, test_classes))
  expect_true(all(ens$agreement))
  expect_equal(ens$prediction, d$label)
  expect_length(mock_log$calls, 3)

  out <- capture.output(sc <- classify_self_consistency(d$text, test_classes, n_runs = 3))
  expect_equal(sc$prediction, d$label)
  expect_equal(unique(sc$confidence), 1)
  expect_equal(dim(attr(sc, "votes")), c(12, 3))
  expect_equal(mock_log$calls[[4]]$temperature, 0.8)
})

test_that("llm_generate and generate_labeled_texts", {
  local_mock_groq()
  expect_equal(as.character(llm_generate("Hello")), "Some generated prose.")
  out <- capture.output(syn <- generate_labeled_texts(c(A = "about A", B = "about B"),
                                                      n_per_topic = 1, n_batches = 2))
  expect_equal(nrow(syn), 4)
  expect_equal(syn$doc_id, 1:4)
  expect_setequal(unique(syn$subtopic), c("A", "B"))
  expect_true(grepl("A: about A", mock_log$calls[[2]]$prompt))
})

test_that("functions give a clear error without a Groq key", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$groq_client
  pkg$groq_client <- NULL
  withr::defer(pkg$groq_client <- old)
  withr::local_envvar(GROQ_API_KEY = "")
  expect_error(classify_direct("a text", "A"), "Every chunk failed.*Groq API key not set")
  expect_error(suppressWarnings(classify_direct(paste("text", 1:3), "A", chunk_size = 1)),
               "Every chunk failed")
  expect_error(llm_generate("x"), "Groq API key not set")
})

test_that("factor classes, malformed entries and edge cases", {
  local_mock_groq()
  d <- make_texts(2)
  res <- classify_scores(d$text, factor(test_classes))
  expect_equal(res$prediction, d$label)
  expect_named(res, c("text_id", "prediction", test_classes))
  out <- capture.output(sc <- classify_self_consistency(d$text, factor(test_classes),
                                                        n_runs = 2, seed = NULL))
  expect_equal(sc$prediction, d$label)
  out <- capture.output(one <- classify_self_consistency(c("x", "y"), "Friendly", n_runs = 2))
  expect_equal(one$Friendly, c(1, 1))

  # A bare string where an object was expected only affects that text
  results <- list("1" = "Friendly", "2" = list(category = "Positive", reasoning = c("a", "b")))
  expect_true(is.na(llmClassificR:::.get_string(results, 1, "category")))
  expect_equal(llmClassificR:::.get_string(results, 2, "reasoning"), "a; b")
  expect_true(is.na(llmClassificR:::.get_string(c("Friendly", "Positive"), 1)))
})

test_that("llm_generate passes extra API arguments", {
  local_mock_groq()
  llm_generate("Search the web", model = "openai/gpt-oss-120b",
               tools = list(list(type = "browser_search")), tool_choice = "required",
               reasoning_effort = "medium")
  call <- mock_log$calls[[1]]
  expect_equal(call$reasoning_effort, "medium")   # user value wins
})

test_that("<think> blocks are removed from answers", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$groq_client
  pkg$groq_client <- list(chat = list(completions = list(create = function(...) {
    list(choices = list(list(message = list(content = "<think>\nhmm\n</think>\nHello"))))
  })))
  withr::defer(pkg$groq_client <- old)
  expect_equal(as.character(llm_generate("Hi")), "Hello")
})
