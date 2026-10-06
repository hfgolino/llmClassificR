test_that("question builders validate their input", {
  q <- jev_choice("Which?", c(A = "desc A", B = "desc B"))
  expect_equal(q$type, "choice")
  expect_equal(q$criteria, list(A = "desc A", B = "desc B"))
  expect_named(jev_choice("Which?", c("A", "B"))$criteria, c("A", "B"))
  expect_error(jev_choice("Which?", "A"), "between 2 and 255")
  expect_error(jev_choice("", c("A", "B")), "instructions")
  expect_null(jev_noul("Is it?")$criteria)
  expect_equal(jev_noul("Is it?", "yes", "no")$criteria, list(true = "yes", false = "no"))
  expect_error(jev_noul("Is it?", true = "yes"), "both")
  expect_equal(jev_score("How much?", c("low", "high"))$criteria, list("low", "high"))
  expect_error(jev_score("How much?", as.character(1:11)), "between 2 and 10")
})

test_that("parsers handle failed requests and missing answers", {
  fake <- list(list(answers = list(
    facet = list(choice = "A", confidence = 0.8, probabilities = list(A = 0.8, B = 0.2)),
    yes = list(noul = 0.9),
    level = list(score = 1.2, confidence = 0.5, probabilities = list("0" = 0.2, "1" = 0.4, "2" = 0.4))
  ), usage = list(input_tokens = 100)), NULL)
  ch <- jev_parse_choice(fake, "facet")
  expect_equal(ch$choice, c("A", NA))
  expect_equal(ch$A, c(0.8, NA))
  expect_equal(jev_parse_noul(fake, "yes")$yes, c(0.9, NA))
  sc <- jev_parse_score(fake, "level")
  expect_equal(sc$level, c(1L, NA))
  expect_named(sc, c("score", "level", "confidence", "0", "1", "2"))
  expect_error(jev_parse_choice(list(NULL), "facet"), "No successful")
  u <- jev_usage(fake)
  expect_equal(u$n_failed, 1)
  expect_equal(u$input_tokens, 100)
  expect_equal(u$cost_usd, round(100 / 1e6 * 0.042, 6))
  expect_equal(jev_usage(fake[[1]])$n_texts, 1)
})

test_that("jev_classify, jev_multilabel and jev_rate work end to end (mocked API)", {
  local_mock_jev()
  d <- make_texts(3)
  crit <- c(Friendly = "warm", Positive = "upbeat", Assertive = "leads", Energetic = "lively")
  out <- capture.output(res <- jev_classify(d$text, crit, max_active = 4))
  expect_equal(res$prediction, d$label)
  expect_named(res, c("text_id", "text", "prediction", "confidence", names(crit)))
  expect_equal(attr(res, "usage")$input_tokens, 300 * 12)
  ev <- evaluate_classification(d$label, res$prediction)
  expect_equal(ev$accuracy, 1)
  cm <- calibration_metrics(res[, names(crit)], d$label)
  expect_equal(cm$mean_confidence, 0.9)

  st <- c(Friendly = "The writer is Friendly.", Positive = "The writer is Positive.")
  out <- capture.output(ml <- jev_multilabel(d$text, st))
  expect_named(ml, c("text_id", "text", "Friendly", "Positive"))
  expect_equal(roc_auc(ml$Friendly, d$label == "Friendly"), 1)

  out <- capture.output(rt <- jev_rate(d$text[1:2], c("low", "mid", "high")))
  expect_equal(rt$score, c(1, 1))
  expect_named(rt, c("text_id", "text", "score", "level", "confidence", "0", "1", "2"))

  one <- jev_ask("A Friendly text", list(f = jev_choice("Which?", crit)))
  expect_equal(one$answers$f$choice, "Friendly")

  models <- jev_list_models()
  expect_equal(models$model, "jev-1.13.0")
})

test_that("failed Jev requests become NA instead of stopping the batch", {
  local_mock_jev()
  d <- make_texts(2)
  d$text[3] <- paste(d$text[3], "FAIL")
  out <- capture.output(res <- jev_classify(d$text, c("Friendly", "Positive", "Assertive", "Energetic")))
  expect_true(is.na(res$prediction[3]))
  expect_equal(sum(is.na(res$prediction)), 1)
  expect_equal(attr(res, "usage")$n_failed, 1)
  expect_error(jev_ask("FAIL", list(f = jev_choice("Which?", c("A", "B")))), "401|API key")
})

test_that("jev functions require named questions and a key", {
  local_mock_jev()
  expect_error(jev_ask("x", list(jev_choice("Which?", c("A", "B")))), "NAMED")
  pkg <- asNamespace("llmClassificR")$.pkg_env
  pkg$typesafe_key <- NULL
  withr::local_envvar(TYPESAFE_API_KEY = "")
  expect_error(jev_ask("x", list(a = jev_noul("Is it?"))), "TypeSafe API key not set")
})
