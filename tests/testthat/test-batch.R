test_that("normalize_labels maps case and punctuation, rejects others", {
  cl <- c("Friendly", "Positive")
  expect_equal(normalize_labels(c(" friendly.", "POSITIVE", "Friendliness", NA), cl),
               c("Friendly", "Positive", NA, NA))
  # Non-Latin scripts and labels that differ only by punctuation
  zh <- c("\u5feb\u4e50", "\u60b2\u4f24")
  expect_equal(normalize_labels(c(zh[2], zh[1], "x", ""), zh), c(zh[2], zh[1], NA, NA))
  expect_equal(normalize_labels(c("C", "C++", " c "), c("C++", "C")), c("C", "C++", NA))
  expect_equal(normalize_labels(c("Não", " nao"), c("Não", "Sim")), c("Não", NA))
})

test_that("run_in_chunks returns the original order and survives failed chunks", {
  texts <- paste("t", 1:10)
  res <- run_in_chunks(texts, function(x) data.frame(echo = x), chunk_size = 3,
                       pause = 0, verbose = FALSE)
  expect_equal(res$echo, texts)
  expect_equal(res$row, 1:10)

  failing <- function(x) {
    if ("t 5" %in% x) stop("boom")
    data.frame(echo = x)
  }
  expect_warning(res2 <- run_in_chunks(texts, failing, chunk_size = 3, pause = 0,
                                       verbose = FALSE), "failed")
  expect_error(run_in_chunks(texts, function(x) stop("no key"), chunk_size = 3, pause = 0,
                             verbose = FALSE), "Every chunk failed. First error: no key")
  expect_equal(nrow(res2), 10)
  expect_true(is.na(res2$echo[5]))
  expect_equal(sum(is.na(res2$echo)), 3)
})

test_that("shuffling restores the global random state", {
  set.seed(42)
  a <- runif(1)
  set.seed(42)
  invisible(run_in_chunks(letters, function(x) data.frame(x = x), chunk_size = 5,
                          pause = 0, verbose = FALSE))
  b <- runif(1)
  expect_equal(a, b)
})

test_that("classification_schema restricts categories", {
  s <- classification_schema(c("A", "B"))
  items <- s$properties$classifications$items
  expect_equal(unlist(items$properties$category$enum), c("A", "B"))
  expect_false(items$additionalProperties)
  expect_null(classification_schema("A", with_confidence = FALSE)$properties$
                classifications$items$properties$confidence)
})
