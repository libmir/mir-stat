"""Print independently generated Fp Poisson PMF reference tests.

Uses Python Decimal at two precisions and evaluates the exact represented
dyadic rate. Logarithmic evaluation avoids constructing enormous powers,
factorials, or probabilities below Decimal's exponent range.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_HALF_EVEN
from math import factorial


def generate(precision):
    rows = []
    with localcontext() as ctx:
        ctx.prec = precision
        ln2 = D(2).ln()

        def rounded(value, bits):
            exponent = int((value.ln() / ln2).to_integral_value(rounding=ROUND_FLOOR)) - bits + 1
            coefficient = int((value / D(2) ** exponent).to_integral_value(rounding=ROUND_HALF_EVEN))
            if coefficient == 1 << bits:
                coefficient >>= 1
                exponent += 1
            return coefficient, exponent

        for bits in (64, 128, 256):
            cases = [(0, D(1)), (1, D(1)), (100, D(1)),
                     (1000, D(1000)), (3000, D(3000)),
                     (0, D(1) + D(2) ** -60),
                     (0, D(2) ** -100), (1, D(2) ** -100),
                     (1, D(2) ** -1100), (2, D(2) ** -1100),
                     # The polynomial factor is exactly halfway at 64 bits;
                     # exp(-rate) puts the true mass below that midpoint.
                     (3, D(3900009) * D(2) ** -1200),
                     (0, D('0.5')), (0, D('0.5') - D(2) ** -60),
                     (0, D('0.5') + D(2) ** -60),
                     (0, D(2) ** 30), (0, D(2) ** 60),
                     (1, D(2) ** 60), (0, D(2) ** 62),
                     (10, D(2) ** 62), (0, D(2) ** 66)]
            # Probe normalization and rounding close to the smallest result.
            for offset in (bits + 2, bits, bits // 2, 1, 0, -1, -2):
                cases.append((0, (D(2) ** 63 - offset) * ln2))
            # exp(-rate) alone underflows, but the final mass is representable.
            cases.append((12, (D(2) ** 63 + 200) * ln2))
            for k, requested_rate in cases:
                rate_coefficient, rate_exponent = rounded(requested_rate, bits)
                rate = D(rate_coefficient) * D(2) ** rate_exponent
                logp = -rate + k * rate.ln() - D(factorial(k)).ln()
                exponent = max(-(1 << 63), int((logp / ln2).to_integral_value(rounding=ROUND_FLOOR)) - bits + 1)
                scaled_log = logp - exponent * ln2
                coefficient = 0 if scaled_log < -ln2 else int(scaled_log.exp().to_integral_value(rounding=ROUND_HALF_EVEN))
                if coefficient == 1 << bits:
                    coefficient >>= 1
                    exponent += 1
                rows.append((bits, k, rate_exponent, rate_coefficient, exponent, coefficient))
    return rows


def render(rows):
    lines = ['// BEGIN Poisson references (tools/generate_fp_poisson_reference.py)',
             'version(mir_stat_test)', '@safe pure nothrow @nogc unittest', '{',
             '    import mir.bignum.fixed: UInt;']
    for bits in (64, 128, 256):
        lines += ['    {', f'        alias T = Fp!{bits};',
                  '        struct Reference { size_t k; T rate, expected; }',
                  '        enum references = [']
        for width, k, re, rc, pe, pc in rows:
            if width == bits:
                exponent = 'long.min' if pe == -(1 << 63) else str(pe) + 'L'
                lines.append(f'            Reference({k}, T(false, {re}, UInt!{bits}.fromHexString("{rc:0{bits//4}x}")), T(false, {exponent}, UInt!{bits}.fromHexString("{pc:0{bits//4}x}"))),')
        lines += ['        ];', '        foreach (c; references)', '        {',
                  '            const actual = poissonPMF(c.k, c.rate);',
                  '            assert(actual.exponent == c.expected.exponent || !actual.coefficient && !c.expected.coefficient);',
                  '            assert(actual.coefficient == c.expected.coefficient);',
                  '            assert(!actual.sign);', '        }', '    }']
    return '\n'.join(lines + ['}', '// END Poisson references'])


if __name__ == '__main__':
    rows = generate(500)
    assert rows == generate(600), 'Reference precision did not stabilize'
    print(render(rows))
