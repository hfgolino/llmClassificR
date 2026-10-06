#' Classify texts using Exploratory Graph Analysis
#'
#' Runs EGA (from the EGAnet package) on an embedding matrix to discover
#' communities (clusters) of texts. Texts are treated as variables and
#' embedding dimensions as observations (transposed), so that correlated
#' texts end up in the same community.
#'
#' @param embeddings A numeric matrix of embeddings (rows = texts, cols = dimensions),
#'   as returned by \code{\link{embed_voyage}} or \code{\link{embed_local}}.
#' @param model Network estimation model: \code{"tmfg"} (default) or \code{"glasso"}.
#' @param corr Correlation method: \code{"pearson"} (default), \code{"spearman"}.
#' @param uni_method Method for unidimensionality: \code{"expand"} (default) or
#'   \code{"LE"}.
#' @param optimize_steps Logical. If \code{TRUE} (default is \code{FALSE}), uses
#'   \code{EGA.fit} to search for the optimal number of walktrap steps (2-25),
#'   which can improve community detection but takes longer. \code{EGA.fit}
#'   chooses the steps with the lowest Total Entropy Fit Index (TEFI), so the
#'   true labels are never used.
#' @param steps_range Integer vector of walktrap steps to try when
#'   \code{optimize_steps = TRUE}. Default is \code{2:25}.
#' @param plot Logical. Whether to plot the EGA network. Default is \code{TRUE}.
#'
#' @return A list with:
#' \describe{
#'   \item{communities}{Integer vector of community assignments (one per text)}
#'   \item{n_communities}{Number of communities found}
#'   \item{TEFI}{Total Entropy Fit Index}
#'   \item{ega_object}{The full EGA object from EGAnet}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' data(extraversion)
#' embeddings <- embed_voyage(extraversion$text)
#'
#' # Basic EGA
#' result <- ega_classify(embeddings)
#' table(result$communities)
#'
#' # Compare with true labels (texts left unassigned count as disagreements)
#' compute_nmi(extraversion$subtopic, result$communities)
#'
#' # With walktrap optimization (chosen by TEFI, not by the labels)
#' result_opt <- ega_classify(embeddings, optimize_steps = TRUE)
#' compute_nmi(extraversion$subtopic, result_opt$communities)
#'
#' # Supervised counterpart: a linear head on the same embeddings
#' linear_probe(embeddings, extraversion$subtopic)
#' }
#'
#' @seealso \code{\link{prime_weight_encode}}, \code{\link{embed_voyage}},
#'   \code{\link{compute_nmi}}, \code{\link{probe_layers}}
#'
#' @export
ega_classify <- function(embeddings,
                         model = "tmfg",
                         corr = "pearson",
                         uni_method = "expand",
                         optimize_steps = FALSE,
                         steps_range = 2:25,
                         plot = TRUE) {

  if (!requireNamespace("EGAnet", quietly = TRUE)) {
    stop("Package 'EGAnet' is required. Install with: install.packages('EGAnet')",
         call. = FALSE)
  }

  # Transpose: EGA expects dimensions as rows, texts as columns
  transposed <- t(embeddings)
  colnames(transposed) <- paste0("doc_", seq_len(ncol(transposed)))
  transposed_df <- as.data.frame(transposed)

  if (optimize_steps) {
    cat("Running EGA.fit with walktrap optimization...\n")
    ega_result <- EGAnet::EGA.fit(
      transposed_df,
      model = model,
      corr = corr,
      uni.method = uni_method,
      steps = steps_range
    )
    ega_obj <- ega_result$EGA
  } else {
    ega_obj <- EGAnet::EGA(
      transposed_df,
      model = model,
      corr = corr,
      uni.method = uni_method,
      plot.EGA = plot
    )
  }

  communities <- as.vector(ega_obj$wc)

  list(
    communities = communities,
    n_communities = length(unique(communities[!is.na(communities)])),
    TEFI = ega_obj$TEFI,
    ega_object = ega_obj
  )
}


#' Prime-weight encoding for multi-layer embeddings
#'
#' Combines EGA networks from multiple embedding layers using prime number
#' weighting. Each layer's EGA network is raised to a prime power and summed,
#' creating a single encoded network that preserves information across layers.
#' Community detection is then performed on the combined network.
#'
#' This technique was introduced by Golino et al. for combining information
#' across transformer layers in a principled way.
#'
#' @param layers A list of embedding matrices (one per layer), as returned
#'   by \code{embed_local()$layers}.
#' @param model Network estimation model for EGA. Default is \code{"tmfg"}.
#' @param uni_method Unidimensionality method for EGA. Default is \code{"expand"}.
#' @param steps_range Integer vector of walktrap steps to search. Default
#'   is \code{2:25}.
#' @param true_labels Optional vector of true labels. If given, the NMI of
#'   every walktrap step is added to \code{step_results} (column \code{NMI})
#'   for teaching and diagnostics. The chosen step never uses the labels.
#'
#' @return A list with:
#' \describe{
#'   \item{best_communities}{Community assignments at the chosen walktrap step}
#'   \item{best_step}{Chosen walktrap step: the first step giving the most
#'     frequent number of communities across \code{steps_range} (label-free)}
#'   \item{n_communities}{Number of communities at the chosen step}
#'   \item{step_results}{Data.frame with the number of communities (and the
#'     NMI, if \code{true_labels} was given) per step}
#'   \item{all_communities}{List of community assignments for every step}
#'   \item{encoding_matrix}{The combined prime-weight encoded network}
#'   \item{layer_egas}{List of per-layer EGA results}
#' }
#'
#' @section Caution:
#' Picking the step with the highest NMI (\code{step_results$NMI}) uses the
#' true labels, so that NMI is an optimistic upper bound, not an
#' unsupervised result. Report the label-free \code{best_communities}.
#'
#' @details
#' The algorithm:
#' \enumerate{
#'   \item Run EGA on each layer separately
#'   \item Extract network adjacency matrices
#'   \item Order networks by edge count (ascending)
#'   \item Assign prime numbers: 2, 3, 5, 7, 11, ...
#'   \item Compute: weight_i = prime_i ^ network_i
#'   \item Sum all weighted networks
#'   \item Apply walktrap community detection, searching for optimal steps
#' }
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' emb <- embed_local(extraversion$text, model_name = "bert-base-uncased")
#'
#' # Prime-weight encoding across all BERT layers
#' pw <- prime_weight_encode(emb$layers)
#'
#' # Best communities
#' table(pw$best_communities)
#'
#' # Compare with true labels
#' compute_nmi(extraversion$subtopic, pw$best_communities)
#'
#' # NMI of every walktrap step (diagnostic only: uses the labels)
#' pw2 <- prime_weight_encode(emb$layers, true_labels = extraversion$subtopic)
#' pw2$step_results
#'
#' # Add Voyage AI embeddings as extra "layer"
#' voyage_emb <- embed_voyage(extraversion$text)
#' combined_layers <- c(emb$layers, list(voyage = voyage_emb))
#' pw_combined <- prime_weight_encode(combined_layers)
#' }
#'
#' @seealso \code{\link{ega_classify}}, \code{\link{embed_local}}
#'
#' @export
prime_weight_encode <- function(layers,
                                model = "tmfg",
                                uni_method = "expand",
                                steps_range = 2:25,
                                true_labels = NULL) {

  if (!requireNamespace("EGAnet", quietly = TRUE)) {
    stop("Package 'EGAnet' is required. Install with: install.packages('EGAnet')",
         call. = FALSE)
  }

  n_layers <- length(layers)
  cat("Running EGA on", n_layers, "layers...\n")

  # Run EGA per layer
  layer_egas <- list()
  for (i in seq_len(n_layers)) {
    cat("  Layer", i, "of", n_layers, "...")
    transposed <- t(layers[[i]])
    colnames(transposed) <- paste0("doc_", seq_len(ncol(transposed)))
    transposed_df <- as.data.frame(transposed)

    layer_egas[[i]] <- tryCatch({
      EGAnet::EGA(transposed_df, model = model, uni.method = uni_method,
                  plot.EGA = FALSE)
    }, error = function(e) {
      cat(" FAILED\n")
      return(NULL)
    })
    cat(" done\n")
  }

  # Remove failed layers
  valid <- !sapply(layer_egas, is.null)
  if (sum(valid) == 0) stop("All layers failed EGA.", call. = FALSE)
  layer_egas <- layer_egas[valid]

  # Extract networks
  individual_networks <- lapply(layer_egas, function(x) x$network)
  individual_sequence <- seq_along(individual_networks)

  # Order by edge count
  edge_counts <- sapply(individual_networks, function(net) {
    sum(abs(net) > 0) / 2
  })
  individual_networks <- individual_networks[order(edge_counts)]

  # Prime number weighting
  prime_numbers <- get(
    utils::data("prime.num", package = "EGAnet", envir = environment())
  )[individual_sequence]

  prime_weights <- lapply(individual_sequence, function(case) {
    prime_numbers[[case]] ^ individual_networks[[case]]
  })

  encoding_matrix <- Reduce("+", prime_weights)

  # Search optimal walktrap steps
  step_results <- data.frame(steps = integer(), n_communities = integer())
  all_communities <- list()

  for (step in steps_range) {
    wc <- do.call(
      what = EGAnet::community.detection,
      args = list(
        network = encoding_matrix,
        algorithm = "walktrap",
        membership.only = TRUE,
        steps = step
      )
    )
    all_communities[[as.character(step)]] <- as.vector(wc)
    step_results <- rbind(step_results, data.frame(
      steps = step,
      n_communities = length(unique(wc[!is.na(wc)]))
    ))
  }
  if (!is.null(true_labels)) {
    step_results$NMI <- vapply(all_communities, function(wc) {
      compute_nmi(true_labels, wc)
    }, numeric(1))
  }

  # Use the step with the most common number of communities
  mode_n <- as.numeric(names(sort(table(step_results$n_communities), decreasing = TRUE))[1])
  best_step <- step_results$steps[step_results$n_communities == mode_n][1]
  best_communities <- all_communities[[as.character(best_step)]]

  cat("Best walktrap step:", best_step, "->", mode_n, "communities\n")

  list(
    best_communities = best_communities,
    best_step = best_step,
    n_communities = mode_n,
    step_results = step_results,
    all_communities = all_communities,
    encoding_matrix = encoding_matrix,
    layer_egas = layer_egas
  )
}
