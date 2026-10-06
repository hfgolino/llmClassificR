test_that("compute_nmi matches aricode::NMI with factors", {
  skip_if_not_installed("aricode")
  set.seed(1)
  for (i in 1:20) {
    x <- sample(letters[1:4], 60, TRUE)
    y <- sample(1:5, 60, TRUE)
    expect_equal(compute_nmi(x, y), aricode::NMI(factor(x), factor(y)), tolerance = 1e-10)
  }
})

test_that("compute_nmi handles label names, NA and edge cases", {
  truth <- rep(c("A", "B", "C"), each = 4)
  expect_equal(compute_nmi(truth, rep(c(3, 1, 2), each = 4)), 1)
  pred <- rep(c(3, 1, 2), each = 4)
  pred[12] <- NA
  nmi_single <- compute_nmi(truth, pred, na = "singleton")
  expect_true(nmi_single < 1 && nmi_single > 0.8)
  expect_equal(compute_nmi(truth, pred, na = "drop"), 1)
  expect_equal(compute_nmi(truth, rep("A", 12)), 0)
  expect_true(is.nan(compute_nmi(rep("A", 3), rep("B", 3))))
  expect_error(compute_nmi(1:3, 1:2), "same length")
})

test_that("evaluate_classification normalizes labels and counts invalid answers", {
  truth <- c("Friendly", "Positive", "Assertive", "Energetic")
  ev <- evaluate_classification(truth, c(" friendly.", "POSITIVE", NA, "Sociable"), "m")
  expect_named(ev, c("method", "accuracy", "macro_F1", "kappa", "NMI", "n_invalid"))
  expect_equal(ev$accuracy, 0.5)
  expect_equal(ev$n_invalid, 2)
  perfect <- evaluate_classification(truth, truth)
  expect_equal(perfect$accuracy, 1)
  expect_equal(perfect$kappa, 1)
  expect_equal(perfect$NMI, 1)
  # data.frame input with a prediction column
  expect_equal(evaluate_classification(truth, data.frame(prediction = truth))$accuracy, 1)
})

test_that("compare_methods works with lowercase predictions and clusters", {
  truth <- rep(c("Friendly", "Positive", "Assertive"), each = 4)
  out <- capture.output(res <- compare_methods(truth,
                                                "lower" = tolower(truth),
                                                "clusters" = rep(c(2, 3, 1), each = 4)))
  expect_equal(res$accuracy[res$method == "lower"], 1)
  expect_equal(res$NMI[res$method == "clusters"], 1)
  expect_equal(res$n_invalid[res$method == "clusters"], 12)
  expect_error(compare_methods(truth, tolower(truth)), "named")
  # A method that failed completely must not beat a random one
  out <- capture.output(res2 <- compare_methods(truth, "failed" = rep(NA, 12),
                                                 "rotated" = truth[c(2:12, 1)]))
  expect_equal(res2$NMI[res2$method == "failed"], 0)
  expect_equal(res2$method[1], "rotated")
})

test_that("roc_auc is correct", {
  expect_equal(roc_auc(c(0.9, 0.8, 0.3, 0.2), c(TRUE, TRUE, FALSE, FALSE)), 1)
  expect_equal(roc_auc(c(0.9, 0.2, 0.8, 0.3), c(TRUE, TRUE, FALSE, FALSE)), 0.5)
  expect_equal(roc_auc(c(0.9, NA, 0.1), c(TRUE, TRUE, FALSE)), 1)
  expect_true(is.na(roc_auc(c(0.1, 0.2), c(TRUE, TRUE))))
})
