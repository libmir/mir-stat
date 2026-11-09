"""Generate Fp hypergeometric log-mass references with mpmath."""
from math import comb
from fractions import Fraction

import mpmath as mp


def reference(k, N, K, n, bits, digits):
    with mp.workdps(digits):
        if min(n, N - n) <= 4096:
            probability = Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))
            # Retain small negative results when the exact probability is near one.
            if probability > Fraction(1, 2):
                value = -mp.log1p(-mp.mpf(probability.denominator - probability.numerator)
                                  / probability.denominator)
            else:
                value = -mp.log(mp.mpf(probability.numerator) / probability.denominator)
        else:
            def log_choose(a, b):
                return mp.loggamma(a+1) - mp.loggamma(b+1) - mp.loggamma(a-b+1)
            value = log_choose(N, n) - log_choose(K, k) - log_choose(N-K, n-k)
        exponent = int(mp.floor(mp.log(value, 2))) - bits + 1
        coefficient = int(mp.nint(mp.ldexp(value, -exponent)))
        if coefficient == 1 << bits:
            exponent += 1
            coefficient >>= 1
        return coefficient, exponent


def render():
    max32 = 2**32 - 1
    # These boundary cases must run in the ordinary suite on either count width.
    boundary32 = [(0, max32, 1, 1), (max32//4, max32, max32//2, max32//2)]
    maximum = 2**64 - 1
    cases = [(0, maximum, 1, 1), (1, maximum, 1, 1),
             (0, maximum, 67, 65), (0, maximum, 1, maximum-1),
             (maximum-2, maximum, maximum-1, maximum-1),
             (maximum//4, maximum, maximum//2, maximum//2),
             (0, maximum, maximum//2, maximum//2),
             (2, 12, 5, 6), (3, 7, 5, 5), (4, 10, 4, 7)] + boundary32
    lines = ['// BEGIN Fp hypergeometric log references']
    # Keep broad low-width coverage and representative public cases at every
    # width in ordinary CI. Run the remaining wider sweep in optimized CI.
    ordinary = {*boundary32, (0, maximum, 1, 1), (maximum//4, maximum, maximum//2, maximum//2), (2, 12, 5, 6)}
    for bits in (64, 128, 256, 512):
        for slow in (False, True):
            selected = [row for row in cases if (bits > 64 and row not in ordinary) == slow]
            if not selected:
                continue
            flag = 'mir_stat_test_slow_numerics' if slow else 'mir_stat_test'
            lines += [f'version({flag})', '@safe pure nothrow @nogc unittest',
                      '{', '    import mir.bignum.fixed: UInt;']
            lines += [f'    alias T = Fp!{bits};',
                      '    struct Reference { ulong k, N, K, n; T value; }',
                      '    enum references = [']
            for row in selected:
                c, e = reference(*row, bits, 500)
                assert (c, e) == reference(*row, bits, 650)
                counts = ', '.join(f'{v}UL' for v in row)
                lines.append(f'        Reference({counts}, T(true, {e}, UInt!{bits}.fromHexString("{c:x}"))),')
            lines += ['    ];', '    foreach (r; references) {',
                      '        if (r.N > size_t.max) continue;',
                      '        assert(hypergeometricLPMF!T(cast(size_t)r.k, cast(size_t)r.N,',
                      '            cast(size_t)r.K, cast(size_t)r.n) == r.value);',
                      '    }', '}']
    return '\n'.join(lines + ['// END Fp hypergeometric log references'])


if __name__ == '__main__':
    print(render())
