from funcs import compare_individual_analyses, compare_bayesian_analyses, compare_bayesian_spline_analyses, status, generate_bayesian_diagnostics

compare_bayesian_analyses("outputs/bayes_standard/")
compare_individual_analyses("outputs/mixed_effects/")
# compare_bayesian_spline_analyses("outputs/bayes_splines/")
# generate_bayesian_diagnostics("outputs/bayes_splines/")
