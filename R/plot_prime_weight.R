#' Match discovered communities to true labels
#'
#' Community numbers are arbitrary (community 1 has nothing to do with the
#' first label). \code{match_communities()} renames each community after the
#' label it overlaps most, using the Hungarian algorithm (Kuhn, 1955) on the
#' community-by-label table: each community gets a different label, and the
#' number of texts whose community name equals their true label is as large
#' as possible.
#'
#' @param found Vector of community assignments (one per text), e.g.,
#'   \code{pw$best_communities}. \code{NA} stays \code{NA}.
#' @param truth Vector of true labels (one per text).
#' @param extra Prefix for communities left over when there are more
#'   communities than labels. Default \code{"Extra"} gives, e.g.,
#'   \code{"Extra 5"}.
#'
#' @return A character vector of matched labels, one per text. When there are
#'   fewer communities than labels, some labels are not used.
#'   \code{mean(matched == truth)} is the matched accuracy: the share of texts
#'   whose community was matched to their own label.
#'
#' @references Kuhn, H. W. (1955). The Hungarian method for the assignment
#'   problem. \emph{Naval Research Logistics Quarterly, 2}(1-2), 83-97.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' found <- c(2, 2, 2, 1, 1, 3, 3, 3)
#' truth <- c("A", "A", "B", "B", "B", "C", "C", "C")
#' matched <- match_communities(found, truth)
#' table(matched, truth)
#' mean(matched == truth)
#'
#' # More communities than labels: the leftover is "Extra 4"
#' match_communities(c(1, 1, 2, 2, 3, 4), c("A", "A", "B", "B", "C", "C"))
#'
#' @seealso \code{\link{plot_prime_weight}}, \code{\link{compute_nmi}}
#'
#' @export
match_communities <- function(found, truth, extra = "Extra") {
  if (length(found) != length(truth)) {
    stop("'found' and 'truth' must have the same length.", call. = FALSE)
  }
  truth <- as.character(truth)
  tab <- unclass(table(found, truth))      # rows = communities, columns = labels
  if (nrow(tab) == 0 || ncol(tab) == 0) {
    stop("No text has both a community and a label.", call. = FALSE)
  }
  tab <- matrix(tab, nrow(tab), ncol(tab), dimnames = dimnames(tab))

  if (nrow(tab) <= ncol(tab)) {
    # Each community gets a different label
    a <- clue::solve_LSAP(tab, maximum = TRUE)
    map <- stats::setNames(colnames(tab)[a], rownames(tab))
  } else {
    # More communities than labels: each label gets a different community
    a <- clue::solve_LSAP(t(tab), maximum = TRUE)
    map <- stats::setNames(paste(extra, rownames(tab)), rownames(tab))
    map[a] <- colnames(tab)
  }
  unname(map[as.character(found)])
}


#' Plot a prime-weight encoding (optionally next to the true labels)
#'
#' Draws the floor-free prime-weight network (\code{pw$network}, values 0-1)
#' with EGAnet, colored by the communities. If \code{true_labels} is given,
#' the communities are first renamed after the labels they overlap most
#' (\code{\link{match_communities}}), and a second panel shows the same network
#' colored by the true labels. Both panels share one layout and one color per
#' label, so a node keeps its color across panels when its community matches
#' its label, and texts that change color are the disagreements.
#'
#' @param pw Output of \code{\link{prime_weight_encode}}. Objects from
#'   llmClassificR 0.2.1 or earlier (without \code{pw$network}) also work.
#' @param true_labels Optional vector of true labels (one per text).
#' @param communities Community assignments to plot. Default
#'   \code{pw$best_communities}. Use, e.g., \code{pw$all_communities[["5"]]}
#'   for another walktrap step.
#' @param titles Panel titles. Default
#'   \code{c("Prime-weight communities", "True labels")}.
#' @param nrow Number of rows when there are two panels. Default \code{2}
#'   (stacked); use \code{1} for side by side.
#' @param palette Optional colors. Either a named vector (names = labels) or
#'   an unnamed vector used in order: the true labels (sorted), then any
#'   extra communities. Default: the first colors of the EGAnet palette.
#' @param seed Seed for the layout. Default \code{2026}. The global random
#'   state is restored afterwards. \code{NULL} leaves the seed alone.
#' @param ... Further arguments for EGAnet's plot (e.g., \code{node.size},
#'   \code{label.size}, \code{edge.size}, \code{mode}). Defaults here:
#'   \code{node.size = 3}, \code{label.size = 0.01} (labels hidden),
#'   \code{edge.size = 0.5}.
#'
#' @return A ggplot object (one panel), or a ggarrange object (two panels).
#'
#' @details
#' The plot always shows the floor-free network, whichever network
#' \code{prime_weight_encode()} used for walktrap (\code{pw$remove_floor}):
#' the raw encoding gives every pair of texts an edge of at least
#' \eqn{L / \sum_k p_k}, which would draw a complete graph. Only pairs linked
#' in at least one layer are drawn.
#'
#' When \code{true_labels} is given, the matched accuracy (the share of texts
#' with the same color in both panels) is shown as a message.
#'
#' @author Hudson Golino \email{hfg9s@@virginia.edu}
#' @examples
#' \dontrun{
#' data(personality_descriptions)
#' traits <- personality_descriptions$Personality
#' emb <- embed_local(personality_descriptions$text, model_name = "bert-base-uncased")
#' pw <- prime_weight_encode(emb$layers, true_labels = traits)
#'
#' # Communities only
#' plot_prime_weight(pw)
#'
#' # Communities and true labels, same layout and matched colors
#' p <- plot_prime_weight(pw, true_labels = traits)
#' p
#' ggplot2::ggsave("prime_weight.png", p, width = 8, height = 12, bg = "white")
#'
#' # Side by side, another walktrap step
#' plot_prime_weight(pw, true_labels = traits, nrow = 1,
#'                   communities = pw$all_communities[["5"]])
#'
#' # The matched labels behind the colors
#' matched <- match_communities(pw$best_communities, traits)
#' table(matched, traits)
#' }
#'
#' @seealso \code{\link{prime_weight_encode}}, \code{\link{match_communities}}
#'
#' @export
plot_prime_weight <- function(pw,
                              true_labels = NULL,
                              communities = NULL,
                              titles = c("Prime-weight communities", "True labels"),
                              nrow = 2,
                              palette = NULL,
                              seed = 2026,
                              ...) {

  if (!requireNamespace("EGAnet", quietly = TRUE)) {
    stop("Package 'EGAnet' is required. Install with: install.packages('EGAnet')",
         call. = FALSE)
  }
  if (is.null(pw$encoding_matrix) || is.null(pw$layer_egas)) {
    stop("'pw' must be the output of prime_weight_encode().", call. = FALSE)
  }

  panels <- .prime_weight_panels(pw, true_labels = true_labels,
                                 communities = communities, titles = titles,
                                 palette = palette, seed = seed, ...)

  if (length(panels$plots) == 1) {
    return(panels$plots[[1]])
  }

  message(sprintf("Matched accuracy: %.1f%% of texts have the same color in both panels.",
                  100 * panels$agreement))

  if (!requireNamespace("ggpubr", quietly = TRUE)) {
    stop("Package 'ggpubr' is required for two panels. Install with: ",
         "install.packages('ggpubr')", call. = FALSE)
  }
  ggpubr::ggarrange(plotlist = panels$plots, nrow = nrow,
                    ncol = ceiling(2 / nrow))
}


# Default colors: the first colors of EGAnet's "polychrome" palette
.pw_default_colors <- c("#F03D2D", "#90DDF0", "#C8D96F", "#EF8A17", "#F5C900",
                        "#BA42C0", "#17BEBB", "#9BAFD9", "#F27A7D", "#F9C58D",
                        "#A18DCE", "#F492F0", "#919BFF", "#FF4F79", "#53917E",
                        "#006BA6")


# Builds the panels (internal; returned as a list for testing)
.prime_weight_panels <- function(pw, true_labels = NULL, communities = NULL,
                                 titles = c("Prime-weight communities", "True labels"),
                                 palette = NULL, seed = 2026, ...) {

  # 1. Floor-free network (computed for objects from older versions)
  R <- pw$network
  if (is.null(R)) {
    primes <- EGAnet::prime.num[seq_along(pw$layer_egas)]
    R <- .prime_floor_free(pw$encoding_matrix, primes)
  }
  n <- ncol(R)

  if (is.null(communities)) communities <- pw$best_communities
  if (length(communities) != n) {
    stop("'communities' must have one value per text (", n, ").", call. = FALSE)
  }

  # 2. Labels: matched to the true labels when given
  if (!is.null(true_labels)) {
    if (length(true_labels) != n) {
      stop("'true_labels' must have one value per text (", n, ").", call. = FALSE)
    }
    true_lab  <- as.character(true_labels)
    found_lab <- match_communities(communities, true_lab)
    lvls <- c(sort(unique(stats::na.omit(true_lab))),
              sort(setdiff(stats::na.omit(found_lab), true_lab)))
  } else {
    true_lab  <- NULL
    found_lab <- ifelse(is.na(communities), NA_character_,
                        paste("Community", communities))
    ids  <- sort(unique(stats::na.omit(communities)))
    lvls <- paste("Community", ids)
  }

  # 3. One color per label, shared by both panels
  pal <- .pw_palette(palette, lvls)

  # 4. EGA template with the floor-free network
  template <- pw$layer_egas[[length(pw$layer_egas)]]
  template$network <- R

  # Plot arguments: defaults, overridden by the user's
  args <- list(...)
  defaults <- list(node.size = 3, label.size = 0.01, edge.size = 0.5)
  if ("vsize" %in% names(args)) defaults$node.size <- NULL
  args <- utils::modifyList(defaults, args)
  args$title <- NULL                 # panel titles come from 'titles'

  # 5. One layout, computed once in the original node order
  #    (a constant wc keeps the order: EGAnet sorts nodes by order(wc))
  coords <- args$mode
  if (is.null(coords) || is.character(coords)) {
    layout_obj <- template
    layout_obj$wc <- stats::setNames(rep(1, n), colnames(R))
    coords <- .with_seed(seed, suppressMessages(suppressWarnings(
      do.call(graphics::plot, c(list(layout_obj, arguments = TRUE), args))$ARGS$mode
    )))
  }
  args$mode <- NULL

  # 6. Panels
  plots <- list(.pw_plot_aligned(template, found_lab, coords, pal, titles[1], args))
  if (!is.null(true_lab)) {
    title2 <- if (length(titles) >= 2) titles[2] else "True labels"
    plots[[2]] <- .pw_plot_aligned(template, true_lab, coords, pal, title2, args)
  }

  list(plots = plots, labels = found_lab, palette = pal, layout = coords,
       agreement = if (is.null(true_lab)) NA_real_ else
         mean(found_lab == true_lab, na.rm = TRUE))
}


# EGAnet's plot sorts the nodes by order(wc) before drawing, so the colors
# and the layout coordinates must be passed in that same order (internal)
.pw_plot_aligned <- function(template, labels, coords, pal, title, args) {
  ega <- template
  ega$wc <- stats::setNames(labels, colnames(ega$network))
  o <- order(labels)
  cols <- unname(pal[labels[o]])
  cols[is.na(cols)] <- "#BFBFBF"
  # node.color must be unnamed: named colors mislabel EGAnet's legend
  suppressMessages(suppressWarnings(
    do.call(graphics::plot, c(list(ega, mode = coords[o, , drop = FALSE],
                                   node.color = cols, title = title), args))
  ))
}


# Named color vector for the labels (internal)
.pw_palette <- function(palette, lvls) {
  if (!is.null(palette) && !is.null(names(palette))) {
    missing <- setdiff(lvls, names(palette))
    if (length(missing) > 0) {
      stop("'palette' has no color for: ", paste(missing, collapse = ", "),
           call. = FALSE)
    }
    return(palette[lvls])
  }
  cols <- if (is.null(palette)) .pw_default_colors else palette
  if (length(cols) < length(lvls)) {
    if (!is.null(palette)) {
      stop("'palette' needs at least ", length(lvls), " colors.", call. = FALSE)
    }
    cols <- c(cols, grDevices::hcl.colors(length(lvls) - length(cols), "Dark 3"))
  }
  stats::setNames(cols[seq_along(lvls)], lvls)
}
