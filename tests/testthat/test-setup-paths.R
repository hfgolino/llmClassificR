test_that("environment names resolve to the working directory or the user data directory", {
  tmp <- withr::local_tempdir()
  withr::local_dir(tmp)
  user_dir <- tools::R_user_dir("llmClassificR", which = "data")

  # No environment in the working directory: the user data directory
  expect_equal(llmClassificR:::.env_dir("llm_env"), file.path(user_dir, "llm_env"))

  # An environment created by earlier versions in the working directory is found first
  dir.create("llm_env")
  expect_equal(normalizePath(llmClassificR:::.env_dir("llm_env")),
               normalizePath(file.path(tmp, "llm_env")))

  # setup_llm_env() never targets the working directory for a name
  expect_equal(llmClassificR:::.env_dir("llm_env", for_setup = TRUE),
               file.path(user_dir, "llm_env"))

  # A path is used as given
  p <- file.path(tmp, "envs", "my_env")
  expect_equal(llmClassificR:::.env_dir(p), p)
  expect_true(startsWith(llmClassificR:::.get_python_path(p), p))
})

test_that("activate_llm_env() explains where it looked when no environment exists", {
  tmp <- withr::local_tempdir()
  withr::local_dir(tmp)
  expect_error(activate_llm_env(file.path(tmp, "missing_env")), "not found")
})
