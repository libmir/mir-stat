"""Generate public Fp negative-binomial LPMF references using mpmath 1.3.0."""
from math import comb

import mpmath as mp


def reference(k,r,c,e,bits,precision):
    with mp.workdps(precision):
        p = mp.ldexp(mp.mpf(c),e)
        # Use exact coefficients when the shorter product is cheap.
        if min(k,r-1) <= 100:
            coefficient = mp.log(comb(k+r-1,min(k,r-1)))
        else:
            coefficient = mp.loggamma(k+r)-mp.loggamma(k+1)-mp.loggamma(r)
        value = -(coefficient + r*mp.log(p) + k*mp.log1p(-p))
        exponent = max(-2**63, int(mp.floor(mp.log(value,2)))-bits+1)
        scaled = mp.ldexp(value,-exponent)
        low = int(mp.floor(scaled))
        remainder = scaled-low
        c = low+(remainder>mp.mpf('0.5') or (remainder==mp.mpf('0.5') and low%2))
        if c == 2**bits:
            c //= 2
            exponent += 1
        return c,exponent


def render():
    lines=['// BEGIN public negative binomial log references','version(mir_stat_test)',
           '@safe pure nothrow @nogc unittest','{','    import mir.bignum.fixed: UInt;']
    for bits in (64,128,256,512):
        maximum = 2**64-1
        cases = [(0,3,1,-2**63), (0,maximum,2**bits-1,-bits),
                 (2,3,1,-2**63), (57,19,3,-4), (1,maximum,2**64-1,-64),
                 (maximum,maximum,1,-1), (maximum-1,maximum,1,-1),
                 (maximum,2,1,-64), (7,12,2**bits-1,-bits)]
        lines+=['    {',f'        alias T = Fp!{bits};',
                '        struct Reference { ulong k; ulong successes; T p; T value; }',
                '        enum references = [']
        for k,n,c,e in sorted(cases):
            r=reference(k,n,c,e,bits,500)
            assert r==reference(k,n,c,e,bits,600)
            out,exp=r
            ep='long.min' if e==-2**63 else str(e)
            ex='long.min' if exp==-2**63 else str(exp)
            lines.append(f'            Reference({k}UL,{n}UL,T(false,{ep},UInt!{bits}.fromHexString("{c:x}")),T(true,{ex},UInt!{bits}.fromHexString("{out:x}"))),')
        lines+=['        ];','        foreach (r; references) {',
                '            if (r.k > size_t.max || r.successes > size_t.max) continue;',
                '            assert(negativeBinomialLPMF(cast(size_t)r.k,cast(size_t)r.successes,r.p) == r.value);',
                '        }','    }']
    return '\n'.join(lines+['}','// END public negative binomial log references'])


if __name__=='__main__':
    print(render())
