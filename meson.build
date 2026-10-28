project('mir-stat', 'd', version : '0.3.1', license: 'Apache-2.0')

description = 'Dlang Statistical Package'

# Match the default DUB library configuration. Optional BLAS and benchmark
# dependencies are exercised by their separate DUB configurations.
subprojects = ['mir-core', 'mir-algorithm']

has_cpp_headers = false

sources_list = [
    'mir/math/internal/benchmark',
    'mir/math/internal/linearAlgebra',
    'mir/math/internal/log1p',
    'mir/math/internal/fp_powi',
    'mir/math/internal/powi',
    'mir/math/internal/log_beta',
    'mir/math/internal/log_binomial',
    'mir/math/internal/xlogy',
    'mir/stat/descriptive/multivariate',
    'mir/stat/descriptive/univariate/package',
    'mir/stat/descriptive/univariate/internal/types',
    'mir/stat/descriptive/univariate/means',
    'mir/stat/descriptive/univariate/sums',
    'mir/stat/descriptive/univariate/quantiles',
    'mir/stat/descriptive/univariate/internal/summation',
    'mir/stat/descriptive/univariate/dispersions',
    'mir/stat/descriptive/univariate/moments',
    'mir/stat/descriptive/univariate/skewness_kurtosis',
    'mir/stat/descriptive/univariate/entropies',
    'mir/stat/descriptive/histogram/breaks',
    'mir/stat/descriptive/histogram/axis',
    'mir/stat/descriptive/histogram/traits',
    'mir/stat/descriptive/histogram/accumulator',
    'mir/stat/descriptive/histogram/relative_frequency',
    'mir/stat/descriptive/histogram/internal/view',
    'mir/stat/descriptive/histogram/internal/density',
    'mir/stat/descriptive/histogram/internal/projection',
    'mir/stat/descriptive/histogram/internal/cell',
    'mir/stat/descriptive/histogram/internal/shared_counts',
    'mir/stat/internal/borrow',
    'mir/stat/internal/fp_probability',
    'mir/stat/internal/fp_geometric_inverse',
    'mir/stat/internal/fp_poisson',
    'mir/stat/internal/fp_scaled_probability',
    'mir/stat/internal/gamma',
    'mir/stat/internal/neg_log1m',
    'mir/stat/internal/one_minus_exp',
    'mir/stat/internal/shape_transform',
    'mir/stat/internal/standardize',
    'mir/stat/internal/summation',
    'mir/stat/descriptive/histogram/package',
    'mir/stat/descriptive/histogram/api/rc',
    'mir/stat/descriptive/histogram/api/gc',
    'mir/stat/descriptive/histogram/api/custom',
    'mir/stat/descriptive/histogram/api/factory',
    'mir/stat/descriptive/histogram/api/package',
    'mir/stat/descriptive/weighted',
    'mir/stat/descriptive/aliases',
    'mir/stat/descriptive/package',
    'mir/stat/constant',
    'mir/stat/inference',
    'mir/stat/transform',
    'mir/stat/package',
    'mir/stat/distribution/bernoulli',
    'mir/stat/distribution/beta',
    'mir/stat/distribution/beta_proportion',
    'mir/stat/distribution/binomial',
    'mir/stat/distribution/categorical',
    'mir/stat/distribution/cauchy',
    'mir/stat/distribution/chi2',
    'mir/stat/distribution/cornish_fisher',
    'mir/stat/distribution/exponential',
    'mir/stat/distribution/f',
    'mir/stat/distribution/gamma',
    'mir/stat/distribution/generalized_pareto',
    'mir/stat/distribution/geometric',
    'mir/stat/distribution/gev',
    'mir/stat/distribution/hypergeometric',
    'mir/stat/distribution/laplace',
    'mir/stat/distribution/log_normal',
    'mir/stat/distribution/logistic',
    'mir/stat/distribution/negative_binomial',
    'mir/stat/distribution/normal',
    'mir/stat/distribution/pareto',
    'mir/stat/distribution/poisson',
    'mir/stat/distribution/rayleigh',
    'mir/stat/distribution/students_t',
    'mir/stat/distribution/uniform',
    'mir/stat/distribution/uniform_discrete',
    'mir/stat/distribution/weibull',
    'mir/stat/distribution/cdf',
    'mir/stat/distribution/invcdf',
    'mir/stat/distribution/pdf',
    'mir/stat/distribution/package',
]

sources = []
foreach s : sources_list
    sources += 'source/' + s + '.d'
endforeach

# Meson interprets -lowmem as a linker library (-l...). Use LDC's long
# spelling so its front-end garbage collector is actually enabled.
lowmem = meson.get_compiler('d').get_id() == 'llvm' ? '--lowmem' : '-lowmem'
add_project_arguments([
    '-preview=dip1008',
    lowmem,
], language: 'd')

required_deps = []

foreach p : subprojects 
    required_deps += dependency(p, fallback : [p, 'this_dep'])
endforeach

directories = ['source']

if has_cpp_headers
    directories += 'include'
endif

directories = include_directories(directories)

this_lib = library(meson.project_name(),
    sources,
    include_directories: directories,
    install: true,
    version: meson.project_version(),
    dependencies: required_deps,
)

this_dep = declare_dependency(
    link_with: [this_lib],
    include_directories: directories,
    dependencies: required_deps,
)

test_versions = ['mir_stat_test']

if has_cpp_headers
    install_subdir('include/',
        strip_directory :true,
        install_dir: 'include/',
    )
endif

install_subdir('source/',
    strip_directory : true,
    install_dir: 'include/d/' + meson.project_name(),
)

import('pkgconfig').generate(this_lib,
    description: description,
    subdirs: 'd/' + meson.project_name(),
)

test_subdirs = []
