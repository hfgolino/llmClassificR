#' Retrieve similar documents
#'
#' Given a query and a matrix of document embeddings, embeds the query and
#' returns the most similar documents by cosine similarity. This is the
#' retrieval step of RAG.
#'
#' @param query A character string (the search query).
#' @param doc_embeddings A numeric matrix of document embeddings (rows = docs),
#'   as returned by \code{\link{embed_voyage}}.
#' @param texts Character vector of original texts (same order as embeddings).
#' @param top_k Number of documents to retrieve. Default is \code{10}.
#'
#' @return A list with:
#' \describe{
#'   \item{indices}{Integer vector of top-k document indices}
#'   \item{texts}{Character vector of retrieved document texts}
#'   \item{scores}{Numeric vector of cosine similarity scores}
#'   \item{all_similarities}{Numeric vector of similarities for all documents}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#' embeddings <- embed_voyage(extraversion$text)
#'
#' # Find texts most related to "assertive leadership"
#' results <- retrieve_similar("assertive leadership", embeddings, extraversion$text)
#' results$texts[1:3]  # Top 3 matches
#' results$scores[1:3] # Their similarity scores
#' }
#'
#' @seealso \code{\link{rag_generate}}, \code{\link{discover_topics}}
#'
#' @export
retrieve_similar <- function(query, doc_embeddings, texts, top_k = 10) {

  vo <- .get_voyageai()

  # Embed query (single API call)
  query_result <- vo$embed(
    texts = list(query),
    model = "voyage-3.5",
    input_type = "query"
  )
  query_vec <- unlist(query_result$embeddings[[1]])

  # Cosine similarity against all documents
  similarities <- apply(doc_embeddings, 1, function(doc_vec) {
    cosine_similarity(query_vec, doc_vec)
  })

  top_indices <- order(similarities, decreasing = TRUE)[
    seq_len(min(top_k, length(similarities)))
  ]

  list(
    indices = top_indices,
    texts = texts[top_indices],
    scores = similarities[top_indices],
    all_similarities = similarities
  )
}


#' Generate an answer using retrieved context (RAG)
#'
#' Retrieves relevant documents for a query, then sends them as context to
#' an LLM to generate a grounded answer. This is the full RAG pipeline.
#'
#' @param query A character string (the question to answer).
#' @param doc_embeddings A numeric matrix of document embeddings.
#' @param texts Character vector of original texts.
#' @param top_k Number of documents to retrieve as context. Default is \code{10}.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#' @param max_tokens Maximum response length. Default is \code{500}.
#'
#' @return A list with:
#' \describe{
#'   \item{answer}{The generated answer}
#'   \item{retrieved}{The retrieved document texts used as context}
#'   \item{scores}{Cosine similarity scores of retrieved documents}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#' @examples
#' \dontrun{
#' data(extraversion)
#' embeddings <- embed_voyage(extraversion$text)
#'
#' result <- rag_generate(
#'   query = "What personality themes are present in these texts?",
#'   doc_embeddings = embeddings,
#'   texts = extraversion$text,
#'   top_k = 15
#' )
#' cat(result$answer)
#' }
#'
#' @seealso \code{\link{retrieve_similar}}, \code{\link{discover_topics}}
#'
#' @export
rag_generate <- function(query, doc_embeddings, texts,
                         top_k = 10,
                         model = "llama-3.3-70b-versatile",
                         max_tokens = 500L) {

  client <- .get_groq_client()

  # Retrieve
  retrieved <- retrieve_similar(query, doc_embeddings, texts, top_k = top_k)

  # Build context
  context <- paste(sapply(seq_along(retrieved$texts), function(i) {
    sprintf("[Document %d]: %s", i, retrieved$texts[i])
  }), collapse = "\n\n")

  prompt <- sprintf(
    'Based on the following documents, answer the query.

DOCUMENTS:
%s

QUERY: %s

Provide a clear, concise answer based on the information in the documents above.',
    context, query
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.2,
    max_tokens = as.integer(max_tokens)
  )

  list(
    answer = response$choices[[1]]$message$content,
    retrieved = retrieved$texts,
    scores = retrieved$scores
  )
}


#' Discover topics automatically via RAG
#'
#' Uses RAG to automatically identify distinct themes or categories in a
#' collection of texts, without any prior labels. The function embeds all
#' documents, retrieves representative samples, and asks an LLM to identify
#' the main themes.
#'
#' @param texts Character vector of texts.
#' @param n_topics Target number of topics to discover. Default is \code{4}.
#' @param embeddings Optional pre-computed embedding matrix. If \code{NULL},
#'   embeddings are computed via \code{\link{embed_voyage}}.
#' @param top_k Number of documents to retrieve for topic analysis. Default
#'   is \code{20}.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A data.frame with columns:
#' \describe{
#'   \item{label}{Short topic label (1-2 words)}
#'   \item{description}{One-sentence description of the topic}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' # Discover topics with no prior labels
#' topics <- discover_topics(extraversion$text, n_topics = 4)
#' print(topics)
#'
#' # Use discovered topics for classification
#' results <- classify_direct(extraversion$text, classes = topics$label)
#' }
#'
#' @seealso \code{\link{classify_with_rag}}, \code{\link{rag_generate}}
#'
#' @export
discover_topics <- function(texts,
                            n_topics = 4,
                            embeddings = NULL,
                            top_k = 20,
                            model = "llama-3.3-70b-versatile") {

  client <- .get_groq_client()

  # Embed if needed
  if (is.null(embeddings)) {
    cat("Embedding documents...\n")
    embeddings <- embed_voyage(texts)
  }

  # Retrieve representative documents
  query <- "What are the main distinct themes or categories present across these texts?"
  retrieved <- retrieve_similar(query, embeddings, texts, top_k = top_k)

  # Ask LLM to identify topics
  prompt <- sprintf(
    'Read these text samples and identify exactly %d distinct themes or categories.

DOCUMENTS:
%s

Return a JSON object with a key "topics" containing an array of objects, each with:
- "label": a short 1-2 word label for the topic
- "description": a one-sentence description
Return ONLY valid JSON.',
    n_topics, paste(retrieved$texts, collapse = "\n---\n")
  )

  response <- client$chat$completions$create(
    model = model,
    messages = list(list("role" = "user", "content" = prompt)),
    temperature = 0.1,
    max_tokens = 400L,
    response_format = list("type" = "json_object")
  )

  result <- tryCatch({
    jsonlite::fromJSON(response$choices[[1]]$message$content)
  }, error = function(e) {
    warning("Failed to parse topics JSON. Returning raw response.")
    return(data.frame(label = NA, description = response$choices[[1]]$message$content))
  })

  topics_df <- as.data.frame(result$topics)
  cat("Discovered", nrow(topics_df), "topics:\n")
  for (i in seq_len(nrow(topics_df))) {
    cat(sprintf("  %d. %s: %s\n", i, topics_df$label[i], topics_df$description[i]))
  }

  topics_df
}


#' Full RAG classification pipeline
#'
#' End-to-end pipeline that embeds documents, discovers topics via RAG, and
#' classifies all texts into the discovered categories. This is a fully
#' unsupervised approach — no prior labels needed.
#'
#' @param texts Character vector of texts.
#' @param n_topics Target number of topics. Default is \code{4}.
#' @param embeddings Optional pre-computed embedding matrix.
#' @param model Groq model to use. Default is \code{"llama-3.3-70b-versatile"}.
#'
#' @return A list with:
#' \describe{
#'   \item{topics}{A data.frame of discovered topics (label + description)}
#'   \item{classifications}{A data.frame of classifications (from \code{\link{classify_direct}})}
#'   \item{embeddings}{The embedding matrix used}
#' }
#'
#' @author Hudson Golino <hfg9s at virginia.edu>
#'
#' @examples
#' \dontrun{
#' data(extraversion)
#'
#' # Full unsupervised pipeline
#' pipeline <- classify_with_rag(extraversion$text, n_topics = 4)
#'
#' # View discovered topics
#' print(pipeline$topics)
#'
#' # View classification results
#' table(pipeline$classifications$prediction)
#'
#' # Compare with true labels (if available)
#' aricode::NMI(extraversion$subtopic, pipeline$classifications$prediction)
#' }
#'
#' @seealso \code{\link{discover_topics}}, \code{\link{classify_direct}}
#'
#' @export
classify_with_rag <- function(texts,
                              n_topics = 4,
                              embeddings = NULL,
                              model = "llama-3.3-70b-versatile") {

  # Step 1: Embed
  if (is.null(embeddings)) {
    cat("Step 1: Embedding documents...\n")
    embeddings <- embed_voyage(texts)
  } else {
    cat("Step 1: Using provided embeddings.\n")
  }

  # Step 2: Discover topics
  cat("Step 2: Discovering topics via RAG...\n")
  topics <- discover_topics(texts, n_topics = n_topics,
                            embeddings = embeddings, model = model)

  # Step 3: Classify
  cat("Step 3: Classifying all texts...\n")
  classifications <- classify_direct(texts, classes = topics$label, model = model)

  list(
    topics = topics,
    classifications = classifications,
    embeddings = embeddings
  )
}
