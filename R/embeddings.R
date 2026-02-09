#' Get Voyage AI embeddings
#'
#' Embeds texts using the Voyage AI API. Returns a numeric matrix where each
#' row is a text and each column is an embedding dimension. Voyage AI's
#' \code{voyage-3.5} model produces 1024-dimensional embeddings.
#'
#' @param texts Character vector of texts to embed.
#' @param model Voyage AI model name. Default is \code{"voyage-3.5"}.
#' @param input_type Either \code{"document"} (for texts to be stored/searched)
#'   or \code{"query"} (for search queries). Default is \code{"document"}.
#'
#' @return A numeric matrix with \code{length(texts)} rows and \code{d}
#'   columns (e.g., 1024 for voyage-3.5).
#'
#' @details
#' Voyage AI's free tier allows 100M tokens/month, which is more than enough
#' for typical course activities. The API supports up to 128 texts per call.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' activate_llm_env()
#' set_voyage_key("your-key")
#'
#' # Embed all documents
#' embeddings <- embed_voyage(extraversion$text)
#' dim(embeddings)  # 120 x 1024
#'
#' # Compute pairwise cosine similarities
#' sim_matrix <- cosine_similarity_matrix(embeddings)
#' }
#'
#' @seealso \code{\link{embed_local}}, \code{\link{retrieve_similar}}
#'
#' @export
embed_voyage <- function(texts,
                         model = "voyage-3.5",
                         input_type = "document") {

  vo <- .get_voyageai()

  result <- vo$embed(
    texts = texts,
    model = model,
    input_type = input_type
  )

  embed_matrix <- do.call(rbind, result$embeddings)
  rownames(embed_matrix) <- paste0("text_", seq_along(texts))
  colnames(embed_matrix) <- paste0("dim_", seq_len(ncol(embed_matrix)))

  embed_matrix
}


#' Get Jina AI embeddings
#'
#' Embeds texts using the Jina AI API. Returns a numeric matrix where each
#' row is a text and each column is an embedding dimension. Jina AI offers
#' several models with different speed/quality tradeoffs and multilingual
#' support. Uses pure R (httr2) with no Python dependency.
#'
#' @param texts Character vector of texts to embed.
#' @param model Jina AI model name. Default is \code{"jina-embeddings-v3"}.
#'   Other options include \code{"jina-embeddings-v4"} (multimodal, 2048-dim),
#'   \code{"jina-embeddings-v2-base-en"} (768-dim), and
#'   \code{"jina-clip-v2"} (text-image, 1024-dim).
#'   See \code{\link{list_jina_models}} for all available models.
#' @param task Task type for the embeddings. Default is \code{"text-matching"}.
#'   Options: \code{"retrieval.passage"} (for documents to be searched),
#'   \code{"retrieval.query"} (for search queries),
#'   \code{"text-matching"} (for similarity comparison),
#'   \code{"classification"} (for classification tasks),
#'   \code{"separation"} (for clustering/separation tasks).
#' @param dimensions Optional integer to reduce embedding dimensions via
#'   Matryoshka Representation Learning (supported by v3 and v4). For example,
#'   \code{dimensions = 256} returns 256-dim instead of the default 1024.
#'   Set to \code{NULL} (default) for full dimensions.
#' @param batch_size Number of texts to send per API request. Default is
#'   \code{100}. Jina supports up to 2048 texts per call, but smaller batches
#'   are more robust.
#'
#' @return A numeric matrix with \code{length(texts)} rows and \code{d}
#'   columns (e.g., 1024 for jina-embeddings-v3).
#'
#' @details
#' Jina AI offers generous free tier limits (1M tokens for free accounts).
#' The v3 model supports 89 languages and Matryoshka dimensions (256, 512,
#' 768, 1024). The API is called via pure R (httr2), so no Python environment
#' is needed for embedding-only workflows.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' set_jina_key("jina_abc123...")
#'
#' # Default: v3, 1024 dimensions
#' embeddings <- embed_jina(extraversion$text)
#' dim(embeddings)  # 120 x 1024
#'
#' # Reduced dimensions (faster, smaller)
#' emb_small <- embed_jina(extraversion$text, dimensions = 256)
#' dim(emb_small)  # 120 x 256
#'
#' # Use for classification with EGA
#' result <- ega_classify(embeddings)
#' }
#'
#' @seealso \code{\link{embed_voyage}}, \code{\link{embed_local}},
#'   \code{\link{list_jina_models}}
#'
#' @export
embed_jina <- function(texts,
                       model = "jina-embeddings-v3",
                       task = "text-matching",
                       dimensions = NULL,
                       batch_size = 100L) {

  api_key <- .get_jina_key()

  n <- length(texts)
  all_embeddings <- list()

  # Process in batches
  n_batches <- ceiling(n / batch_size)
  if (n_batches > 1) {
    cat("Embedding", n, "texts in", n_batches, "batches...\n")
  }

  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1) * batch_size + 1
    end_idx <- min(b * batch_size, n)
    batch_texts <- texts[start_idx:end_idx]

    if (n_batches > 1) {
      cat("  Batch", b, "of", n_batches, "(", length(batch_texts), "texts)...\n")
    }

    result <- .jina_embed_request(
      texts = batch_texts,
      model = model,
      task = task,
      dimensions = dimensions,
      api_key = api_key
    )

    all_embeddings <- c(all_embeddings, result)
  }

  embed_matrix <- do.call(rbind, all_embeddings)
  rownames(embed_matrix) <- paste0("text_", seq_len(n))
  colnames(embed_matrix) <- paste0("dim_", seq_len(ncol(embed_matrix)))

  embed_matrix
}


#' List Jina AI embedding models
#'
#' Returns a data.frame of available Jina AI embedding models with their
#' specifications. No API key needed.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{model}{Model identifier (use this in \code{\link{embed_jina}})}
#'   \item{description}{Model description with size and dimensions}
#'   \item{dimensions}{Default embedding dimensions}
#'   \item{matryoshka}{Whether the model supports dimension reduction}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' list_jina_models()
#'
#' @export
list_jina_models <- function() {
  models <- data.frame(
    model = c(
      "jina-embeddings-v4",
      "jina-embeddings-v3",
      "jina-clip-v2",
      "jina-embeddings-v2-base-en",
      "jina-embeddings-v2-base-zh",
      "jina-embeddings-v2-base-de",
      "jina-embeddings-v2-base-es",
      "jina-embeddings-v2-base-code",
      "jina-embeddings-v2-small-en"
    ),
    description = c(
      "Multimodal flagship (3.8B params, 2048-dim)",
      "Multilingual flagship (570M params, 1024-dim, 89 languages)",
      "Text-image embeddings (885M params, 1024-dim)",
      "English base (137M params, 768-dim)",
      "Chinese base (137M params, 768-dim)",
      "German base (137M params, 768-dim)",
      "Spanish base (137M params, 768-dim)",
      "Code base (137M params, 768-dim)",
      "English small (33M params, 512-dim)"
    ),
    dimensions = c(2048L, 1024L, 1024L, 768L, 768L, 768L, 768L, 768L, 512L),
    matryoshka = c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )

  cat("Available Jina AI embedding models:\n")
  cat("----------------------------------------------------------------------\n")
  for (i in seq_len(nrow(models))) {
    cat(sprintf("  %-35s %s\n", models$model[i], models$description[i]))
  }
  cat("----------------------------------------------------------------------\n")
  invisible(models)
}


#' Make a Jina AI embedding API request (internal)
#' @param texts List or character vector of texts
#' @param model Model name
#' @param task Task type
#' @param dimensions Optional dimension reduction
#' @param api_key API key
#' @return List of numeric vectors (one per text)
#' @noRd
.jina_embed_request <- function(texts, model, task = "text-matching",
                                dimensions = NULL, api_key) {

  if (!requireNamespace("httr2", quietly = TRUE)) {
    stop("Package 'httr2' is required for Jina AI embeddings.\n",
         "Install with: install.packages('httr2')", call. = FALSE)
  }

  # Build request body
  body <- list(
    model = model,
    input = as.list(texts),
    task = task
  )
  if (!is.null(dimensions)) {
    body$dimensions <- as.integer(dimensions)
  }

  # Make request
  resp <- httr2::request("https://api.jina.ai/v1/embeddings") |>
    httr2::req_headers(
      "Authorization" = paste("Bearer", api_key),
      "Content-Type" = "application/json",
      "Accept" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_error(body = function(resp) {
      parsed <- httr2::resp_body_json(resp)
      paste0(parsed$detail %||% "Unknown error")
    }) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)

  # Extract embeddings (returned sorted by index)
  embeddings <- lapply(result$data, function(item) {
    unlist(item$embedding)
  })

  embeddings
}


#' Get local HuggingFace encoder embeddings (all layers)
#'
#' Loads a HuggingFace encoder model (BERT, RoBERTa, DeBERTa, etc.) and
#' extracts mean-pooled embeddings from every hidden layer. This allows you
#' to analyze which layers best capture your text distinctions.
#'
#' @param texts Character vector of texts to embed.
#' @param model_name HuggingFace model identifier. Default is \code{"roberta-base"}.
#'   Other options: \code{"bert-base-uncased"}, \code{"microsoft/deberta-v3-base"},
#'   \code{"albert-base-v2"}, \code{"google/electra-base-discriminator"},
#'   \code{"answerdotai/ModernBERT-base"}.
#' @param max_length Maximum token length for truncation. Default is \code{128}.
#'
#' @return A list with:
#' \describe{
#'   \item{layers}{A list of numeric matrices, one per hidden layer. Each matrix
#'     has \code{length(texts)} rows and \code{hidden_size} columns.}
#'   \item{mean_pooled}{A numeric matrix of embeddings averaged across all layers.}
#'   \item{model_name}{The model name used.}
#'   \item{num_layers}{Number of hidden layers in the model.}
#'   \item{hidden_size}{Dimensionality of each layer.}
#' }
#'
#' @details
#' The function extracts hidden states from all transformer layers, then
#' mean-pools over non-padding tokens to produce one vector per text per layer.
#' This requires PyTorch and transformers to be installed in the environment.
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' # Extract from BERT
#' emb <- embed_local(extraversion$text, model_name = "bert-base-uncased")
#' cat("Layers:", emb$num_layers, "| Hidden size:", emb$hidden_size, "\n")
#'
#' # Embeddings from best layer
#' layer_quality <- evaluate_layers(emb$layers, extraversion$subtopic)
#' best <- which.max(layer_quality$combined_score)
#' best_embeddings <- emb$layers[[best]]
#' }
#'
#' @seealso \code{\link{embed_voyage}}, \code{\link{evaluate_layers}}
#'
#' @export
embed_local <- function(texts,
                        model_name = "roberta-base",
                        max_length = 128L) {

  transformers <- reticulate::import("transformers")
  torch <- reticulate::import("torch")

  cat("Loading model:", model_name, "...\n")
  tokenizer <- transformers$AutoTokenizer$from_pretrained(
    model_name,
    clean_up_tokenization_spaces = TRUE
  )
  model <- transformers$AutoModel$from_pretrained(
    model_name,
    output_hidden_states = TRUE,
    return_dict = TRUE,
    use_safetensors = TRUE
  )
  model$eval()

  num_layers <- model$config$num_hidden_layers
  hidden_size <- model$config$hidden_size
  cat("Layers:", num_layers, "| Hidden size:", hidden_size, "\n")

  all_layer_embeddings <- vector("list", num_layers + 1L)
  for (l in seq_along(all_layer_embeddings)) {
    all_layer_embeddings[[l]] <- matrix(0, nrow = length(texts), ncol = hidden_size)
  }

  cat("Extracting embeddings for", length(texts), "texts...\n")
  for (i in seq_along(texts)) {
    if (i %% 20 == 0) cat("  Processing text", i, "of", length(texts), "\n")

    inputs <- tokenizer(
      texts[i],
      return_tensors = "pt",
      truncation = TRUE,
      padding = "max_length",
      max_length = as.integer(max_length)
    )

    with(torch$no_grad(), {
      outputs <- model(inputs$input_ids, attention_mask = inputs$attention_mask)
    })

    hidden_states <- outputs$hidden_states

    for (layer_idx in seq_along(hidden_states)) {
      layer_output <- hidden_states[[layer_idx]]
      attention_mask <- inputs$attention_mask$unsqueeze(-1L)$expand(layer_output$size())
      masked_output <- layer_output * attention_mask$float()
      summed <- torch$sum(masked_output, dim = 1L)
      counts <- torch$sum(attention_mask$float(), dim = 1L)
      mean_pooled <- summed / counts
      all_layer_embeddings[[layer_idx]][i, ] <- mean_pooled$numpy()[1, ]
    }
  }
  cat("Extraction complete!\n")

  # Name the matrices
  for (l in seq_along(all_layer_embeddings)) {
    rownames(all_layer_embeddings[[l]]) <- paste0("text_", seq_along(texts))
    colnames(all_layer_embeddings[[l]]) <- paste0("dim_", seq_len(hidden_size))
  }
  names(all_layer_embeddings) <- paste0("layer_", seq(0, num_layers))

  # Mean across all layers
  embed_array <- array(
    unlist(all_layer_embeddings),
    dim = c(length(texts), hidden_size, length(all_layer_embeddings))
  )
  mean_pooled <- apply(embed_array, c(1, 2), mean)
  rownames(mean_pooled) <- paste0("text_", seq_along(texts))
  colnames(mean_pooled) <- paste0("dim_", seq_len(hidden_size))

  list(
    layers = all_layer_embeddings,
    mean_pooled = mean_pooled,
    model_name = model_name,
    num_layers = num_layers,
    hidden_size = hidden_size
  )
}


#' Evaluate embedding layer quality
#'
#' Assesses how well each layer's embeddings separate the true text categories
#' using three R-native metrics: silhouette score (via k-means clustering),
#' PCA explained variance, and within-class vs. between-class distance ratio.
#'
#' @param layers A list of embedding matrices (one per layer), as returned
#'   by \code{\link{embed_local}}.
#' @param labels A character or factor vector of true category labels.
#'
#' @return A data.frame with one row per layer and columns:
#' \describe{
#'   \item{layer}{Layer name}
#'   \item{layer_number}{Layer index (0 = embedding layer)}
#'   \item{silhouette}{Mean silhouette score from k-means clustering}
#'   \item{explained_var}{Proportion of variance explained by first 4 PCs}
#'   \item{separation_ratio}{Between-class / within-class distance ratio}
#'   \item{combined_score}{Z-score sum of all three metrics}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' emb <- embed_local(extraversion$text, model_name = "bert-base-uncased")
#'
#' quality <- evaluate_layers(emb$layers, extraversion$subtopic)
#' print(quality)
#'
#' # Best layer
#' best <- quality[which.max(quality$combined_score), ]
#' cat("Best layer:", best$layer, "- Combined score:", best$combined_score, "\n")
#'
#' # Plot
#' plot(quality$layer_number, quality$combined_score, type = "b",
#'      xlab = "Layer", ylab = "Combined Score", main = "Layer Quality")
#' }
#'
#' @seealso \code{\link{embed_local}}
#'
#' @export
evaluate_layers <- function(layers, labels) {

  labels <- as.character(labels)
  k <- length(unique(labels))
  n_layers <- length(layers)

  results <- data.frame(
    layer = names(layers),
    layer_number = seq(0, n_layers - 1),
    silhouette = numeric(n_layers),
    explained_var = numeric(n_layers),
    separation_ratio = numeric(n_layers),
    stringsAsFactors = FALSE
  )

  cat("Evaluating", n_layers, "layers...\n")

  for (l in seq_len(n_layers)) {
    emb <- layers[[l]]

    # 1. K-means + silhouette (R native)
    km <- stats::kmeans(emb, centers = k, nstart = 10)
    d <- stats::dist(emb)
    sil <- cluster::silhouette(km$cluster, d)
    results$silhouette[l] <- mean(sil[, "sil_width"])

    # 2. PCA explained variance
    pca <- stats::prcomp(emb, center = TRUE, scale. = FALSE, rank. = 4)
    results$explained_var[l] <- sum(pca$sdev^2) / sum(apply(emb, 2, stats::var))

    # 3. Within vs between class distance ratio
    unique_labels <- unique(labels)
    within_d <- c()
    between_d <- c()

    for (lab in unique_labels) {
      idx_in <- which(labels == lab)
      idx_out <- which(labels != lab)
      if (length(idx_in) > 1) {
        class_dist <- as.matrix(stats::dist(emb[idx_in, , drop = FALSE]))
        within_d <- c(within_d, class_dist[upper.tri(class_dist)])
      }
      # Sample between-class distances for speed
      n_sample <- min(5, length(idx_in), length(idx_out))
      if (n_sample > 0) {
        for (ci in idx_in[seq_len(n_sample)]) {
          for (oi in idx_out[seq_len(n_sample)]) {
            between_d <- c(between_d, sqrt(sum((emb[ci, ] - emb[oi, ])^2)))
          }
        }
      }
    }

    results$separation_ratio[l] <- mean(between_d) / mean(within_d)
  }

  # Combined z-score
  z <- function(x) (x - mean(x)) / (stats::sd(x) + 1e-10)
  results$combined_score <- z(results$silhouette) +
    z(results$explained_var) +
    z(results$separation_ratio)

  cat("Done.\n")
  results
}
