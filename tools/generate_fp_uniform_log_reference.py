"""Generate stabilized log-mass references and a certified fixed-point log(2)."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import random
import sys


def scaled_log(n, bits, precision):
    with localcontext() as ctx:
        ctx.prec = precision
        x = F(D(n).ln())
    exponent = x.numerator.bit_length() - x.denominator.bit_length()
    if x < F(2) ** exponent:
        exponent -= 1
    exponent -= bits - 1
    return x / F(2) ** exponent, exponent


def reference(n, bits, precision):
    scaled, exponent = scaled_log(n, bits, precision)
    return round(scaled), exponent


def log_two_hex():
    # Certify floor(2^512*log(2)) with exact rational atanh-series bounds.
    lower = sum((F(2, (2*k+1)*3**(2*k+1)) for k in range(200)), F(0))
    tail_bound = F(2, 401*3**401) / (1-F(1,9))
    value = int(lower * 2**512)
    assert value == int((lower+tail_bound) * 2**512)
    for precision in (500, 600):
        with localcontext() as ctx:
            ctx.prec = precision
            assert value == int(D(2).ln() * D(2)**512)
    return f'{value:0128x}'


def sample_counts():
    rng = random.Random(157)
    return sorted({rng.randrange(2, 2**rng.randrange(2,65)+1) for _ in range(2048)})


def selected_counts():
    counts = set(range(2,18))
    counts.update(2**k+d for k in (16,32,53,63,64) for d in (-1,0,1) if 2 <= 2**k+d <= 2**64)
    # Retain the eight closest sampled rounding midpoints for each tested width.
    sample = sample_counts()
    for bits in (64,128,256):
        def distance(n):
            x, _ = scaled_log(n,bits,240)
            return abs(x - int(x) - F(1,2))
        counts.update(sorted(sample, key=distance)[:8])
    return sorted(counts)


def render(counts=None, widths=(64,128,256,512)):
    if counts is None:
        counts = selected_counts()
    lines = ['// BEGIN discrete-uniform log references','version(mir_stat_test)',
             '@safe pure nothrow @nogc unittest','{']
    for bits in widths:
        lines += ['    {',f'        alias T = Fp!{bits};',
                  '        struct Reference { ulong upper; T logMass; }','        enum references = [']
        for n in counts:
            c,e = reference(n,bits,500)
            assert (c,e) == reference(n,bits,600)
            lines.append(f'            Reference({n-1}UL,T(true,{e},UInt!{bits}.fromHexString("{c:x}"))),')
        lines += ['        ];','        foreach (c; references) {',
                  '            if (c.upper > size_t.max) continue;',
                  '            const upper = cast(size_t)c.upper;',
                  '            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);',
                  '            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);',
                  '        }','    }']
    lines += ['}','// END discrete-uniform log references']
    return '\n'.join(lines)


if __name__ == '__main__':
    print(log_two_hex() if '--log-two' in sys.argv else render())
