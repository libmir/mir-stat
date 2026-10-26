"""Print D tests using exact integer binomial masses and ties-to-even rounding.

No floating-point library participates in reference generation. Inputs are
dyadic rationals m/2**scale, so each mass is an integer divided by a power of 2.
Run with Python 3; paste stdout into binomial.d between the reference markers.
"""
from math import comb


def rounded(numerator, scale, bits):
    shift = numerator.bit_length() - bits
    if shift <= 0:
        return numerator << -shift, shift - scale
    quotient, remainder = divmod(numerator, 1 << shift)
    halfway = 1 << (shift - 1)
    quotient += remainder > halfway or (remainder == halfway and quotient & 1)
    if quotient == 1 << bits:
        quotient >>= 1
        shift += 1
    return quotient, shift - scale


print('// BEGIN exact binomial references (tools/generate_fp_binomial_reference.py)')
print('version(mir_stat_test)\n@safe pure nothrow @nogc unittest\n{')
print('    import mir.bignum.fixed: UInt;')
print('    struct Reference { size_t k, n; string input; int scale; string coefficient; long exponent; }')
for bits in (64, 128, 256):
    cases = [
        (17, 70, (1 << bits) // 3, bits),
        (23, 200, 3602879701896397, 55),  # exact represented double 0.1
        (500, 1000, 1, 1),
        (1, 200, (1 << bits) - 1, bits),
        (199, 200, (1 << bits) - 1, bits),
        (0, 10000, 1, bits + 1),  # complement rounds to 1 at output precision
        (0, 1, 1, bits + 1),  # exactly halfway: round to even (1)
        (10000, 10000, (1 << bits) - 1, bits),
        (73, 211, (1 << (bits - 1)) + 12345, bits),
    ]
    print('    {\n        alias T = Fp!%d;' % bits)
    print('        static foreach (c; [')
    rows = []
    for k, n, m, scale in cases:
        numerator = comb(n, k) * pow(m, k) * pow((1 << scale) - m, n - k)
        coefficient, exponent = rounded(numerator, scale * n, bits)
        rows.append('            Reference(%d, %d, "%d", %d, "%d", %d)' %
                    (k, n, m, scale, coefficient, exponent))
    print(',\n'.join(rows))
    print('        ]) {{')
    print('            enum input = UInt!%d(c.input);' % bits)
    print('            enum expected = UInt!%d(c.coefficient);' % bits)
    print('            T p = T(input);')
    print('            p.exponent -= c.scale;')
    print('            const actual = binomialPMF(c.k, c.n, p);')
    print('            assert(!actual.sign);')
    print('            assert(actual.exponent == c.exponent);')
    print('            assert(actual.coefficient == expected);')
    print('        }}\n    }')
print('}\n// END exact binomial references')
