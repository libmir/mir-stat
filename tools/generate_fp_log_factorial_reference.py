"""Generate exact Stirling coefficients and independent log-factorial references.

Reference mode requires mpmath 1.3.0. Coefficient mode uses only the standard
library. Counts and coefficient arithmetic are exact integers/rationals.
"""
from fractions import Fraction as F
from math import comb


def coefficients():
    bernoulli=[F(1)]
    for n in range(1,131):
        bernoulli.append(-sum(F(comb(n+1,k))*bernoulli[k] for k in range(n))/F(n+1))
    return [abs(bernoulli[2*k]/F(2*k*(2*k-1))) for k in range(1,66)]


def render_coefficients():
    lines=['// BEGIN Stirling coefficients',
           'private struct StirlingCoefficient { UInt!512 numerator; UInt!512 denominator; }',
           'private immutable StirlingCoefficient[65] coefficients = [']
    for c in coefficients():
        assert max(c.numerator.bit_length(),c.denominator.bit_length())<=512
        lines.append(f'    StirlingCoefficient(UInt!512.fromHexString("{c.numerator:x}"),UInt!512.fromHexString("{c.denominator:x}")),')
    return '\n'.join(lines+['];','// END Stirling coefficients'])


def crossover(bits):
    c=coefficients()[-1]
    numerator=c.numerator << (bits+320)
    lo,hi=1,2**64-1
    while lo<hi:
        mid=(lo+hi)//2
        if numerator<c.denominator*mid**129: hi=mid
        else: lo=mid+1
    return lo


def reference(n,bits,precision):
    import mpmath as mp
    with mp.workdps(precision):
        x=mp.mpf(n)
        fact=mp.loggamma(x+1)
        corr=fact-(x+mp.mpf('0.5'))*mp.log(x)+x-mp.log(2*mp.pi)/2
        def rounded(value):
            if not value: return 0,0
            exponent=int(mp.floor(mp.log(value,2)))-bits+1
            scaled=mp.ldexp(value,-exponent)
            floor=int(mp.floor(scaled))
            remainder=scaled-floor
            c=floor+(remainder>mp.mpf('0.5') or (remainder==mp.mpf('0.5') and floor%2))
            if c==2**bits: c//=2; exponent+=1
            return c,exponent
        return rounded(fact),rounded(corr)


def render():
    lines = ['// BEGIN log-factorial references']
    for bits in (64,128,256,512):
        block_start = len(lines)
        threshold=crossover(bits)
        counts=sorted({1,2,3,16,32,threshold-1,threshold,threshold+1,2**32-1,2**32,10**12,2**64-1})
        width=bits+192
        lines+=['    {',f'        alias L = CountLog!{bits};',f'        alias W = Fp!{width};',
                f'        assert(!L.usesStirling({threshold-1}));',f'        assert(L.usesStirling({threshold}));',
                '        assert(L.factorial(0) == W(0));',
                '        struct Reference { ulong n; W factorial; W correction; }','        enum references = [']
        for n in counts:
            r=reference(n,width,500)
            assert r==reference(n,width,600)
            def fp(c,e):
                return f'W(false,{e},UInt!{width}.fromHexString("{c:x}"))' if c else 'W(0)'
            lines.append(f'            Reference({n}UL,{fp(*r[0])},{fp(*r[1])}),')
        lines+=['        ];','        foreach (r; references) {',
                '            assert(L.factorial(r.n) == r.factorial);',
                '            assert(L.correction(r.n) == r.correction);','        }','    }']
        block = lines[block_start:]
        version = 'mir_stat_test_slow_numerics' if bits >= 512 else 'mir_stat_test'
        lines[block_start:] = [f'version({version})',
                              '@safe pure nothrow @nogc unittest',
                              *(line[4:] for line in block), '']
    return '\n'.join(lines+['// END log-factorial references'])


if __name__=='__main__':
    import sys
    print(render_coefficients() if '--coefficients' in sys.argv else render())
