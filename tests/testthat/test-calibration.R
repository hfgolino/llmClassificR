test_that("softmax works for vectors and matrices", {
  p <- softmax(c(a = 1, b = 1))
  expect_equal(unname(p), c(0.5, 0.5))
  expect_named(p, c("a", "b"))
  M <- rbind(c(A = 1000, B = 0), c(A = 0, B = 0))
  P <- softmax(M)
  expect_equal(rowSums(P), c(1, 1))
  expect_equal(unname(P[1, "A"]), 1)
  expect_error(softmax(1:3, temperature = 0), "positive")
  P2 <- softmax(data.frame(text_id = 1:2, A = c(2, 0), B = c(0, 2)))
  expect_equal(unname(rowSums(P2)), c(1, 1))
  expect_equal(colnames(P2), c("A", "B"))
})

test_that("ECE and Brier score give known values", {
  expect_equal(expected_calibration_error(rep(0.95, 10), c(rep(TRUE, 6), rep(FALSE, 4))), 0.35)
  expect_equal(expected_calibration_error(c(1, 1, NA), c(TRUE, TRUE, FALSE)), 0)
  probs <- rbind(c(A = 1, B = 0), c(A = 0, B = 1))
  expect_equal(brier_score(probs, c("A", "B")), 0)
  expect_equal(brier_score(probs, c("b", "a")), 2)   # labels normalized
  expect_equal(brier_score(c(1, 0), c(TRUE, FALSE)), 0)
  expect_error(expected_calibration_error(c(90, 80), c(TRUE, FALSE)), "between 0 and 1")
  expect_error(reliability_table(c(90, 80), c(TRUE, FALSE)), "between 0 and 1")
})

test_that("calibration_metrics drops missing rows and reports them", {
  probs <- rbind(c(A = 0.8, B = 0.2), c(A = NA, B = NA), c(A = 30, B = 70))
  cm <- calibration_metrics(probs, c("A", "A", "B"), "test")
  expect_equal(cm$n, 2)
  expect_equal(cm$n_missing, 1)
  expect_equal(cm$accuracy, 1)
  expect_equal(cm$mean_confidence, 0.75)
  # bookkeeping columns of a whole output data.frame are ignored
  df <- data.frame(text_id = 1:2, prediction = c("A", "B"), confidence = c(0.8, 0.7),
                   A = c(0.8, 0.3), B = c(0.2, 0.7))
  expect_equal(calibration_metrics(df, c("A", "B"))$accuracy, 1)
})

test_that("fit_temperature recovers a known temperature", {
  set.seed(2026)
  n <- 3000
  S <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, c("a", "b", "c")))
  P <- softmax(S / 0.5)
  y <- apply(P, 1, function(p) sample(colnames(P), 1, prob = p))
  expect_equal(fit_temperature(S, y), 0.5, tolerance = 0.1)
})

test_that("temperature scaling repairs the offline MiniLM probabilities", {
  data(minilm_embeddings, package = "llmClassificR", envir = environment())
  data(neuroticism, package = "llmClassificR", envir = environment())
  S <- cosine_scores(minilm_embeddings$texts$neuroticism, minilm_embeddings$labels$emotions)
  y <- neuroticism$subtopic
  before <- calibration_metrics(softmax(S), y)
  after <- calibration_metrics(softmax(S, fit_temperature(S, y)), y)
  expect_equal(before$accuracy, after$accuracy)
  expect_lt(after$ECE, before$ECE)
})

test_that("reliability_table accepts matrices and vectors", {
  rel <- reliability_table(c(0.95, 0.95, 0.55), c(TRUE, FALSE, TRUE), "m")
  expect_equal(sum(rel$n), 3)
  expect_equal(rel$accuracy[rel$confidence == 0.95], 0.5)
  probs <- rbind(c(A = 0.9, B = 0.1), c(A = 0.2, B = 0.8))
  rel2 <- reliability_table(probs, c("A", "A"))
  expect_equal(sum(rel2$n), 2)
  skip_if_not_installed("ggplot2")
  expect_s3_class(plot_reliability(rel), "ggplot")
})

test_that("Platt scaling and RLCR reward behave", {
  set.seed(1)
  score <- c(rnorm(100, 1), rnorm(100, -1))
  y <- rep(c(TRUE, FALSE), each = 100)
  fit <- fit_platt(score, y)
  expect_s3_class(fit, "platt_fit")
  expect_gt(fit$a, 0)
  p <- predict(fit, c(-5, 0, 5))
  expect_true(p[1] < p[2] && p[2] < p[3])
  expect_length(predict(fit), 200)
  expect_output(print(fit), "Platt scaling")
  expect_equal(rlcr_reward(0.9, FALSE), -0.81)
  expect_equal(rlcr_reward(0.9, TRUE), 0.99)
  q <- seq(0, 1, 0.01)
  expect_equal(q[which.max(0.7 * rlcr_reward(q, TRUE) + 0.3 * rlcr_reward(q, FALSE))], 0.7)
})
