"""Independent references for guarded discrete probabilities.

Small integer powers use exact Fraction arithmetic; full-width counts use
Decimal at 500 and 600 digits and must produce identical rounded results.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_HALF_EVEN
from fractions import Fraction as F
from math import comb


def rounded(value, bits):
    if not value:
        return "T(0)"
    if isinstance(value, F):
        a, b = value.numerator, value.denominator
        e = a.bit_length() - b.bit_length()
        if (a < b << e) if e >= 0 else (a << -e < b):
            e -= 1
        e -= bits - 1
        c = round(value * F(2) ** -e)
    else:
        e = int((value.ln() / D(2).ln()).to_integral_value(rounding=ROUND_FLOOR)) - bits + 1
        c = int((value / D(2) ** e).to_integral_value(rounding=ROUND_HALF_EVEN))
    if c == 1 << bits:
        c >>= 1
        e += 1
    return f'T(false, {e}L, UInt!{bits}.fromHexString("{c:x}"))'


def generate(precision):
    tails, masses = [], []
    with localcontext() as ctx:
        ctx.prec = precision
        for bits in (64, 128, 256):
            cases = [(0, 1, -1), (1, 1, -2), (17, 3, -2),
                     (1000, 1, -3), (3, 2**bits-1, -bits),
                     (1, 1, -1100), (2, 2**(bits-1)+1, -1100),
                     (2**64-1, 1, -64), (2**64-1, 1, -bits-1),
                     (2**60, 1, -60), (65, 1, -bits)]
            for k, a, e in cases:
                p = F(a) * F(2) ** e
                if k <= 1000:
                    tail = (1-p) ** (k+1)
                    cdf = 1-tail
                else:
                    dp = D(a) * D(2) ** e
                    tail = (D(k+1)*(1-dp).ln()).exp()
                    cdf = 1-tail
                tails.append((bits, k, a, e, rounded(cdf, bits), rounded(tail, bits)))
            cases = [(0, 2, 1, -1), (1, 2, 1, -1), (10, 5, 1, -2),
                     (100, 101, 1, -1), (1000, 3, 1, -4),
                     (1, 2, 2**bits-1, -bits), (2, 3, 1, -1100),
                     (2**64-1, 2, 1, -bits-1),
                     (2**64-1, 3, 1, -64), (1, 2**64-1, 2**bits-1, -bits)]
            for k, r, a, e in cases:
                p = F(a) * F(2) ** e
                coefficient = comb(k+r-1, min(k, r-1))
                if max(k, r) <= 1000:
                    mass = F(coefficient) * p**r * (1-p)**k
                else:
                    dp = D(a) * D(2) ** e
                    mass = (D(coefficient).ln()+D(r)*dp.ln()+D(k)*(1-dp).ln()).exp()
                masses.append((bits, k, r, a, e, rounded(mass, bits)))
    return tails, masses


def probability(a, e, bits):
    return f'T(false, {e-bits+a.bit_length()}, UInt!{bits}.fromHexString("{a << (bits-a.bit_length()):x}"))'


def render(rows, tail):
    label = 'geometric-tail' if tail else 'negative-binomial'
    lines = [f'// BEGIN {label} references', 'version(mir_stat_test)',
             '@safe pure nothrow @nogc unittest', '{', '    import mir.bignum.fixed: UInt;']
    for bits in (64, 128, 256):
        fields = 'ulong k; T p, cdf, ccdf;' if tail else 'ulong k, r; T p, mass;'
        lines += ['    {', f'        alias T = Fp!{bits};',
                  f'        struct Reference {{ {fields} }}', '        enum references = [']
        for row in rows:
            if row[0] != bits:
                continue
            if tail:
                _, k, a, e, cdf, ccdf = row
                line = f'{k}UL, {probability(a,e,bits)}, {cdf}, {ccdf}'
            else:
                _, k, r, a, e, mass = row
                line = f'{k}UL, {r}UL, {probability(a,e,bits)}, {mass}'
            lines.append(f'            Reference({line}),')
        lines += ['        ];', '        foreach (c; references) {']
        if tail:
            lines += ['            if (c.k > size_t.max) continue;',
                      '            assert(geometricCDF(cast(size_t)c.k, c.p) == c.cdf);',
                      '            assert(geometricCCDF(cast(size_t)c.k, c.p) == c.ccdf);']
        else:
            lines += ['            if (c.k > size_t.max || c.r > size_t.max) continue;',
                      '            assert(negativeBinomialPMF(cast(size_t)c.k, cast(size_t)c.r, c.p) == c.mass);']
        lines += ['        }', '    }']
    return '\n'.join(lines + ['}', f'// END {label} references'])


if __name__ == '__main__':
    a = generate(500)
    assert a == generate(600)
    print(render(a[0], True))
    print(render(a[1], False))
