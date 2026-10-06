emb <- function() {
  e <- new.env()
  data(minilm_embeddings, package = "llmClassificR", envir = e)
  e$minilm_embeddings
}

test_that("classify_by_description reproduces the course zero-shot accuracy", {
  e <- emb()
  data(extraversion, package = "llmClassificR", envir = environment())
  res <- classify_by_description(text_embeddings = e$texts$extraversion,
                                 label_embeddings = e$labels$extraversion)
  expect_equal(nrow(res), 120)
  expect_equal(round(mean(res$prediction == extraversion$subtopic), 2), 0.89)
  expect_equal(unname(rowSums(res[, rownames(e$labels$extraversion)])), rep(1, 120))
  expect_equal(dim(attr(res, "scores")), c(120, 4))
  expect_error(classify_by_description(), "texts")
  expect_error(cosine_scores(matrix(1, 2, 3), matrix(1, 2, 4)), "same number")
})

test_that("classify_by_description embeds with a user function", {
  toy_embed <- function(x) {
    t(vapply(x, function(s) c(nchar(s), grepl("a", s), 1), numeric(3)))
  }
  res <- classify_by_description(c("aaa", "bbbbbb"), c(A = "a", B = "bbbbbb"),
                                 embed_fun = toy_embed)
  expect_equal(res$text, c("aaa", "bbbbbb"))
  expect_true(all(res$prediction %in% c("A", "B")))
})

test_that("train_shared_head fits and predicts on a new task", {
  e <- emb()
  data(extraversion, package = "llmClassificR", envir = environment())
  data(neuroticism, package = "llmClassificR", envir = environment())
  tasks <- list(
    list(text_embeddings = e$texts$extraversion, label_embeddings = e$labels$extraversion,
         labels = extraversion$subtopic)
  )
  fit <- train_shared_head(tasks, lambda = 1e-3, maxit = 20)
  expect_s3_class(fit, "shared_head")
  expect_length(fit$w, 384)
  expect_output(print(fit), "Shared scoring head")
  P <- predict(fit, e$texts$neuroticism, e$labels$emotions)
  expect_equal(dim(P), c(120, 4))
  expect_equal(unname(rowSums(P)), rep(1, 120))
  expect_error(train_shared_head(list(list(text_embeddings = 1))), "Each task")
  bad <- tasks
  bad[[1]]$labels[1] <- "Unknown"
  expect_error(train_shared_head(bad), "not row names")
})

test_that("cv_logistic and linear_probe on embeddings", {
  skip_if_not_installed("glmnet")
  e <- emb()
  data(extraversion, package = "llmClassificR", envir = environment())
  cv <- cv_logistic(e$texts$extraversion, extraversion$subtopic)
  expect_equal(nrow(cv), 120)
  expect_equal(sort(unique(cv$fold)), 1:5)
  expect_equal(unname(rowSums(cv[, sort(unique(extraversion$subtopic))])), rep(1, 120),
               tolerance = 1e-8)
  lp <- linear_probe(e$texts$extraversion, extraversion$subtopic)
  expect_gt(lp$accuracy, 0.85)
  # same seed, same result
  expect_equal(cv_logistic(e$texts$extraversion, extraversion$subtopic)$prediction, cv$prediction)
  # whole output can go into calibration_metrics()
  expect_equal(calibration_metrics(cv, cv$truth)$accuracy,
               round(mean(cv$prediction == cv$truth), 3))
  # clear errors for tiny classes and missing labels
  small <- rep(c("A", "B"), each = 3)
  expect_error(cv_logistic(e$texts$extraversion[1:6, ], small), "at least 4 texts")
  y_na <- extraversion$subtopic
  y_na[1] <- NA
  expect_error(cv_logistic(e$texts$extraversion, y_na), "missing values")
  cv5 <- suppressWarnings(cv_logistic(e$texts$extraversion[1:48, ],
                                      rep(c("A", "B", "C", "D"), each = 12), k = 5))
  expect_equal(as.vector(table(cv5$fold)), rep(48 / 5, 5), tolerance = 0.25)
})

test_that("probe_layers works on a list of matrices", {
  skip_if_not_installed("glmnet")
  e <- emb()
  data(extraversion, package = "llmClassificR", envir = environment())
  layers <- list(layer_0 = e$texts$extraversion[, 1:50], layer_1 = e$texts$extraversion)
  out <- capture.output(pl <- probe_layers(layers, extraversion$subtopic))
  expect_equal(pl$layer_number, 0:1)
  names(layers) <- c("layer_3", "layer_9")
  out <- capture.output(pl2 <- probe_layers(layers, extraversion$subtopic))
  expect_equal(pl2$layer_number, c(3, 9))
  expect_true(all(pl$probe_accuracy > 0.5))
})

test_that("bag-of-words baseline, top terms and learning curve", {
  skip_if_not_installed("tm")
  skip_if_not_installed("SnowballC")
  skip_if_not_installed("glmnet")
  data(extraversion, package = "llmClassificR", envir = environment())
  X <- bow_features(extraversion$text)
  expect_s4_class(X, "dgCMatrix")
  expect_equal(nrow(X), 120)
  out <- capture.output(bow <- bow_baseline(extraversion$text, extraversion$subtopic, n_top = 3))
  expect_gt(bow$evaluation$accuracy, 0.9)
  expect_true(all(table(bow$top_terms$class) <= 3))
  lc <- learning_curve(X, extraversion$subtopic, n_per_class = c(3, 12), n_repeats = 2)
  expect_equal(lc$n_train, c(12, 48))
  expect_error(learning_curve(X, extraversion$subtopic, n_per_class = 30), "more texts")
  expect_lt(ncol(bow_features(extraversion$text, min_docs = 3)), ncol(X))
})

test_that("diversity_report summarizes synthetic data", {
  r <- diversity_report(c("I love people", "I love people", "Energy all day long"),
                        c("A", "A", "B"), "toy")
  expect_equal(r$exact_duplicates, 1)
  expect_equal(r$within_label_overlap, 1)
  expect_equal(r$top_opening_share, round(2 / 3, 3))
  r2 <- diversity_report(c("one text", "another"))
  expect_true(is.na(r2$within_label_overlap))
})

test_that("evaluate_layers reports the variance share of the first 4 PCs", {
  e <- emb()
  data(extraversion, package = "llmClassificR", envir = environment())
  out <- capture.output(q <- evaluate_layers(list(layer_0 = e$texts$extraversion),
                                              extraversion$subtopic))
  p <- prcomp(e$texts$extraversion)
  expect_equal(q$explained_var, sum(p$sdev[1:4]^2) / sum(p$sdev^2))
  expect_lt(q$explained_var, 0.9)
})

test_that("embed_local(isolate = TRUE) asks for activate_llm_env() first", {
  pkg <- asNamespace("llmClassificR")$.pkg_env
  old <- pkg$python
  pkg$python <- NULL
  withr::defer(pkg$python <- old)
  expect_error(embed_local("a text", "bert-base-uncased", isolate = TRUE),
               "activate_llm_env")
})
