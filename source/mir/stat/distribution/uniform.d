/++
This module contains algorithms for the uniform continuous probability distribution.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.stat.distribution.uniform;

import mir.internal.utility: isFloatingPoint;

/++
Computes the uniform probability distribution function (PDF).

Params:
    x = value to evaluate PDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformPDF(T)(T x, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(x >= lower, "x must be greater than or equal to lower bound in uniform probability distribution.");
    assert(x <= upper, "x must be less than or equal to upper bound in uniform probability distribution.");
    assert(lower < upper, "lower must be less than upper");

    return 1.0L / (upper - lower);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual;
    assert(0.5.uniformPDF == 1);
    assert(0.5.uniformPDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformPDF(1.0, 3.0).approxEqual(0.5));
}

/++
Computes the uniform cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformCDF(T)(T x, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(x >= lower, "x must be greater than or equal to lower bound in uniform probability distribution.");
    assert(x <= upper, "x must be less than or equal to upper bound in uniform probability distribution.");
    assert(lower < upper, "lower must be less than upper");

    return (x - lower) / (upper - lower);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual;
    assert(0.5.uniformCDF == 0.5);
    assert(0.5.uniformCDF(0.0, 1.5).approxEqual(1.0 / 3));
    assert(2.5.uniformCDF(1.0, 3.0).approxEqual(3.0 / 4));
}

/++
Computes the uniform complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformCCDF(T)(T x, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(x >= lower, "x must be greater than or equal to lower bound in uniform probability distribution.");
    assert(x <= upper, "x must be less than or equal to upper bound in uniform probability distribution.");
    assert(lower < upper, "lower must be less than upper");

    return (upper - x) / (upper - lower);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual;
    assert(0.5.uniformCCDF == 0.5);
    assert(0.5.uniformCCDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformCCDF(1.0, 3.0).approxEqual(1.0 / 4));
}

/++
Computes the uniform inverse cumulative distribution function (InvCDF)

Params:
    p = value to evaluate InvCDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformInvCDF(T)(T p, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(p >= 0, "p must be greater than or equal to 0.");
    assert(p <= 1, "p must be less than or equal to 1.");
    assert(lower < upper, "lower must be less than upper");

    return lower + p * (upper - lower);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual;
    assert(0.5.uniformInvCDF == 0.5);
    assert((1.0 / 3).uniformInvCDF(0.0, 1.5).approxEqual(0.5));
    assert((3.0 / 4).uniformInvCDF(1.0, 3.0).approxEqual(2.5));
}

/++
Computes the uniform log probability distribution function (LPDF)

Params:
    x = value to evaluate LPDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformLPDF(T)(T x, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(x >= lower, "x must be greater than or equal to lower bound in uniform probability distribution.");
    assert(x <= upper, "x must be less than or equal to upper bound in uniform probability distribution.");
    assert(lower < upper, "lower must be less than upper");

    import mir.math.common: log;

    return -log(upper - lower);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual, log;
    assert(0.5.uniformLPDF == 0);
    assert(0.5.uniformLPDF(0.0, 1.5).approxEqual(-log(1.5)));
    assert(1.5.uniformLPDF(1.0, 3.0).approxEqual(-log(2.0)));
}

/++
Computes the uniform log unnormalized probability distribution function (LuPDF)

This removes constants that would typically get dropped in estimation.

Params:
    x = value to evaluate LuPDF
    lower = lower bound
    upper = upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, uniform probability distribution)
+/
T uniformLuPDF(T)(T x, T lower = 0, T upper = 1)
    if (isFloatingPoint!T)
{
    assert(x >= lower, "x must be greater than or equal to lower bound in uniform probability distribution.");
    assert(x <= upper, "x must be less than or equal to upper bound in uniform probability distribution.");
    assert(lower < upper, "lower must be less than upper");

    return cast(T) 1;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest {
    import mir.math.common: approxEqual, log;
    assert(0.5.uniformLuPDF == 1);
    assert(0.5.uniformLuPDF(0.0, 1.5) == 1);
    assert(2.5.uniformLuPDF(1.0, 3.0) == 1);
}