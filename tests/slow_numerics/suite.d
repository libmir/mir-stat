/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.tests.slow_numerics;

// Import the modules with slow tests; the compiler follows implementation imports.
import mir.stat.distribution.binomial;
import mir.stat.distribution.negative_binomial;
import mir.stat.internal.fp_binomial_log;
import mir.stat.internal.fp_log_factorial;
import mir.stat.internal.fp_poisson_log;
