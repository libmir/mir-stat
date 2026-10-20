/++
Compatibility imports for weighted means and sums.

Weighted means and their assumptions now belong to
$(MREF mir,stat,descriptive,univariate,means); weighted sums belong to
$(MREF mir,stat,descriptive,univariate,sums). This module re-exports the same
public declarations for existing callers. Weighted quantiles remain in
$(MREF mir,stat,descriptive,univariate,quantiles).

Import $(MREF mir,stat,descriptive,univariate) for all these statistical families.
Moving the defining modules preserves source imports but changes symbol names
in compiled code; rebuild consumers when upgrading.

$(SCRIPT
(function () {
    var owners = {"wmean": "means", "WMeanAccumulator": "means", "AssumeWeights": "means", "wsum": "sums", "WSummator": "sums"};
    var anchor = window.location.hash.substring(1);
    var symbol = anchor.replace(/^\./, '').split('.')[0];
    if (Object.prototype.hasOwnProperty.call(owners, symbol)) {
        window.location.replace('mir_stat_descriptive_univariate_' + owners[symbol] + '.html#' + anchor);
    }
})();
)

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2022 Mir Stat Authors.
+/
module mir.stat.descriptive.weighted;

///
public import mir.stat.descriptive.univariate.means: wmean, WMeanAccumulator, AssumeWeights;
///
public import mir.stat.descriptive.univariate.sums: wsum, WSummator;

// Preserve identity through legacy, family, and umbrella imports.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import old = mir.stat.descriptive.weighted;
    import means = mir.stat.descriptive.univariate.means;
    import sums = mir.stat.descriptive.univariate.sums;
    import all = mir.stat.descriptive;
    import uni = mir.stat.descriptive.univariate;
    static foreach (name; ["wmean", "WMeanAccumulator", "AssumeWeights"])
    {
        static assert(__traits(isSame, __traits(getMember, old, name), __traits(getMember, means, name)));
        static assert(__traits(isSame, __traits(getMember, all, name), __traits(getMember, means, name)));
        static assert(__traits(isSame, __traits(getMember, uni, name), __traits(getMember, means, name)));
    }
    static foreach (name; ["wsum", "WSummator"])
    {
        static assert(__traits(isSame, __traits(getMember, old, name), __traits(getMember, sums, name)));
        static assert(__traits(isSame, __traits(getMember, all, name), __traits(getMember, sums, name)));
        static assert(__traits(isSame, __traits(getMember, uni, name), __traits(getMember, sums, name)));
    }
    assert(old.wmean([2.0, 8.0], [1.0, 3.0]) == 6.5);
    assert(old.wsum([2.0, 8.0], [1.0, 3.0]) == 26);
}
