"""Generate discrete-uniform Fp references using exact Python rational arithmetic."""
from fractions import Fraction


def rounded(value, bits):
    if not value:
        return "T(0)"
    normalized = value
    exponent = 0
    while normalized < 1:
        normalized *= 2
        exponent -= 1
    exponent -= bits - 1
    coefficient = round(value * (1 << -exponent))
    if coefficient == 1 << bits:
        coefficient >>= 1
        exponent += 1
    error = abs(Fraction(coefficient) * Fraction(2) ** exponent - value)
    assert error <= Fraction(2) ** (exponent - 1)
    return f'T(false, {exponent}, UInt!{bits}.fromHexString("{coefficient:x}"))'


def render():
    maximum = 2**64 - 1
    cases = [(0, 0, 0), (0, 0, 1), (1, 0, 2), (0, 0, 2),
             (2, 0, 2), (4, 0, 6), (7, 0, 9), (13, 10, 26),
             (0, 0, 2**53 - 1), (0, 0, 2**53),
             (2**53 - 1, 0, 2**53), (0, 0, maximum),
             (maximum - 1, 0, maximum), (maximum, 0, maximum),
             (maximum - 1, maximum - 2, maximum),
             (2**63, 0, maximum - 1)]
    lines = ['// BEGIN discrete-uniform references', 'version(mir_stat_test)',
             '@safe pure nothrow @nogc unittest', '{']
    for bits in (64, 128, 256):
        lines += ['    {', f'        alias T = Fp!{bits};',
                  '        struct Reference { ulong x, lower, upper; T pmf, cdf, ccdf; }',
                  '        enum references = [']
        for x, lower, upper in cases:
            n = upper - lower + 1
            expected = [rounded(Fraction(a, n), bits)
                        for a in (1, x - lower + 1, upper - x)]
            lines.append(f'            Reference({x}UL, {lower}UL, {upper}UL, ' + ', '.join(expected) + '),')
        lines += ['        ];', '        foreach (c; references) {',
                  '            if (c.upper > size_t.max) continue;',
                  '            const x = cast(size_t)c.x;',
                  '            const lower = cast(size_t)c.lower;',
                  '            const upper = cast(size_t)c.upper;',
                  '            assert(uniformDiscretePMF!T(x, lower, upper) == c.pmf);',
                  '            assert(uniformDiscreteCDF!T(x, lower, upper) == c.cdf);',
                  '            assert(uniformDiscreteCCDF!T(x, lower, upper) == c.ccdf);',
                  '        }', '    }']
    return '\n'.join(lines + ['}', '// END discrete-uniform references'])


if __name__ == '__main__':
    print(render())
