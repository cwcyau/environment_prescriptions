from funcs import compare_individual_analyses, compare_bayesian_analyses, compare_bayesian_spline_analyses, status

compare_individual_analyses("outputs/mixed_effects/")
compare_bayesian_analyses("outputs/bayes_standard/")
compare_bayesian_spline_analyses("outputs/bayes_splines/")