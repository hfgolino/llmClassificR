# Tests for match_communities(), the floor-free network, and plot_prime_weight()

test_that("match_communities() renames communities after the best labels", {
  found <- c(2, 2, 2, 1, 1, 3, 3, 3)
  truth <- c("A", "A", "B", "B", "B", "C", "C", "C")
  m <- match_communities(found, truth)
  expect_equal(m, c("A", "A", "A", "B", "B", "C", "C", "C"))
  expect_equal(mean(m == truth), 7 / 8)

  # Factor labels work the same way
  expect_equal(match_communities(found, factor(truth)), m)
})

test_that("match_communities() handles more or fewer communities than labels", {
  # More communities: the leftover one is "Extra k"
  m <- match_communities(c(1, 1, 2, 2, 3, 4), c("A", "A", "B", "B", "C", "C"))
  expect_equal(m[1:4], c("A", "A", "B", "B"))
  expect_equal(sum(grepl("^Extra ", m)), 1)
  expect_true(all(m[5:6] %in% c("C", "Extra 3", "Extra 4")))

  # Fewer communities: each gets a different label, one label is unused
  m2 <- match_communities(c(1, 1, 1, 2, 2, 2), c("A", "A", "B", "C", "C", "C"))
  expect_equal(m2, c("A", "A", "A", "C", "C", "C"))

  # Custom prefix and NA communities
  m3 <- match_communities(c(1, 2, 3, NA), c("A", "B", "B", "A"), extra = "New")
  expect_true(is.na(m3[4]))
  expect_equal(sum(grepl("^New ", m3)), 1)

  expect_error(match_communities(1:3, c("A", "B")), "same length")
})

test_that("the floor-free network maps the floor to 0 and the ceiling to 1", {
  primes <- c(2, 3, 5)
  n <- 4
  nets <- list(matrix(0, n, n), matrix(0, n, n), matrix(0, n, n))
  for (k in 1:3) diag(nets[[k]]) <- 1
  # Pair (1, 2) linked with weight 1 in all networks; pair (1, 3) only in network 3
  for (k in 1:3) nets[[k]][1, 2] <- nets[[k]][2, 1] <- 1
  nets[[3]][1, 3] <- nets[[3]][3, 1] <- 0.5
  # Pair (2, 4) has a negative edge in network 1
  nets[[1]][2, 4] <- nets[[1]][4, 2] <- -0.5
  X <- Reduce("+", Map(function(p, A) p^A, primes, nets))

  R <- .prime_floor_free(X, primes)
  expect_equal(R[1, 2], 1)
  expect_equal(R[1, 3], (5^0.5 - 1) / (sum(primes) - 3))
  expect_equal(R[3, 4], 0)                # never linked
  expect_equal(R[2, 4], 0)                # negative: set to 0
  expect_equal(unname(diag(R)), rep(0, n))
  expect_true(isSymmetric(R))
  expect_equal(colnames(R), paste0("doc_", 1:n))
})

# Small synthetic data: 30 texts in 3 clusters, 3 "layers"
.pw_test_data <- function() {
  set.seed(11)
  cl <- rep(1:3, each = 10)
  cent <- matrix(stats::rnorm(3 * 60), 3, 60)
  layers <- lapply(1:3, function(l) {
    cent[cl, ] * (1 + l / 3) + matrix(stats::rnorm(30 * 60, sd = 1.2), 30, 60)
  })
  list(layers = layers, cl = cl)
}

.quiet <- function(expr) {
  out <- NULL
  utils::capture.output(out <- suppressMessages(suppressWarnings(expr)))
  out
}

test_that("prime_weight_encode() returns the floor-free network and keeps old results by default", {
  skip_on_cran()
  skip_if_not_installed("EGAnet")
  dat <- .pw_test_data()

  pw <- .quiet(prime_weight_encode(dat$layers))
  expect_false(pw$remove_floor)
  expect_equal(pw$primes, c(2, 3, 5))
  expect_equal(pw$network, .prime_floor_free(pw$encoding_matrix, pw$primes))
  expect_true(all(pw$network >= 0 & pw$network <= 1))

  # Default: walktrap on the raw encoding, as in 0.2.1
  old <- EGAnet::community.detection(pw$encoding_matrix, algorithm = "walktrap",
                                     steps = pw$best_step, membership.only = TRUE)
  expect_equal(pw$best_communities, as.vector(old))

  # remove_floor = TRUE: walktrap on the floor-free network
  pw_ff <- .quiet(prime_weight_encode(dat$layers, remove_floor = TRUE))
  expect_true(pw_ff$remove_floor)
  ff <- EGAnet::community.detection(pw_ff$network, algorithm = "walktrap",
                                    steps = pw_ff$best_step, membership.only = TRUE)
  expect_equal(pw_ff$best_communities, as.vector(ff))
})

test_that("plot_prime_weight() keeps the layout and the color of each label across panels", {
  skip_on_cran()
  skip_if_not_installed("EGAnet")
  skip_if_not_installed("ggpubr")
  dat <- .pw_test_data()
  pw <- .quiet(prime_weight_encode(dat$layers))

  # True labels with a few disagreements, so the panels differ
  truth <- c("Alpha", "Beta", "Gamma")[dat$cl]
  truth[c(1, 12, 25)] <- c("Gamma", "Alpha", "Beta")

  pn <- .quiet(.prime_weight_panels(pw, true_labels = truth))
  expect_length(pn$plots, 2)
  matched <- match_communities(pw$best_communities, truth)
  expect_equal(pn$labels, matched)
  expect_equal(pn$agreement, mean(matched == truth))

  # Node name, position and color in each panel, in the original node order
  node_data <- function(p) {
    b <- ggplot2::ggplot_build(p)$data
    txt <- b[[which(vapply(b, function(d) "label" %in% names(d), logical(1)))[1]]]
    pts <- b[[length(b) - 2]]           # filled points, same order as the labels
    d <- data.frame(name = txt$label, x = txt$x, y = txt$y, colour = pts$colour)
    d[match(paste0("doc_", seq_along(truth)), d$name), ]
  }
  d1 <- node_data(pn$plots[[1]])
  d2 <- node_data(pn$plots[[2]])

  expect_equal(d1$x, d2$x)
  expect_equal(d1$y, d2$y)
  expect_equal(toupper(d1$colour), toupper(unname(pn$palette[matched])))
  expect_equal(toupper(d2$colour), toupper(unname(pn$palette[truth])))
  # Same color in both panels exactly when community and label agree
  expect_equal(d1$colour == d2$colour, matched == truth)

  # The exported function returns one ggarrange object (two panels) or a ggplot
  expect_true(inherits(.quiet(plot_prime_weight(pw, true_labels = truth)), "ggarrange"))
  expect_true(inherits(.quiet(plot_prime_weight(pw)), "ggplot"))

  # Objects from earlier versions (no $network) still work
  pw_old <- pw[setdiff(names(pw), c("network", "remove_floor", "primes"))]
  pn_old <- .quiet(.prime_weight_panels(pw_old, true_labels = truth))
  expect_equal(node_data(pn_old$plots[[1]])$colour, d1$colour)

  # Named palette and errors
  pal <- c(Alpha = "#000000", Beta = "#444444", Gamma = "#888888")
  pn_pal <- .quiet(.prime_weight_panels(pw, true_labels = truth, palette = pal))
  expect_equal(unname(pn_pal$palette[sort(unique(truth))]), unname(pal))
  expect_error(.prime_weight_panels(pw, true_labels = truth, palette = c(Alpha = "#000000")),
               "no color")
  expect_error(plot_prime_weight(pw, true_labels = truth[-1]), "one value per text")
})
