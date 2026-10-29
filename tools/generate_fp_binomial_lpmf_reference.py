"""Generate public Fp binomial LPMF references using mpmath 1.3.0."""
import mpmath as mp


def reference(k,n,c,e,bits,precision):
    with mp.workdps(precision):
        p = mp.ldexp(mp.mpf(c),e)
        # Endpoint formulas avoid cancellation before subnormal rounding.
        if k == 0:
            value = -n*mp.log1p(-p)
        elif k == n:
            value = -n*(mp.log(c)+e*mp.log(2))
        else:
            value = -(mp.loggamma(n+1)-mp.loggamma(k+1)-mp.loggamma(n-k+1)
                      + k*mp.log(p)+(n-k)*mp.log1p(-p))
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
    lines=['// BEGIN public binomial log references','version(mir_stat_test)',
           '@safe pure nothrow @nogc unittest','{','    import mir.bignum.fixed: UInt;']
    for bits in (64,128,256,512):
        cases=set()
        for n in (1,3,2**64-1):
            for c,e in ((1,-1),(1,-2000),(1,-2**63),(2**bits-1,-bits)):
                cases.add((0,n,c,e))
                cases.add((n,n,c,e))
        cases.add((0,3,2**(bits-1)+3,-2*bits-400))
        cases.update(((1,2,1,-1),(2,7,3,-3),
                      (2**63-1,2**64-2,1,-1), (1,2**64-1,1,-64)))
        lines+=['    {',f'        alias T = Fp!{bits};',
                '        struct Reference { ulong k; ulong n; T p; T value; }',
                '        enum references = [']
        for k,n,c,e in sorted(cases):
            r=reference(k,n,c,e,bits,500)
            assert r==reference(k,n,c,e,bits,600)
            out,exp=r
            ep='long.min' if e==-2**63 else str(e)
            ex='long.min' if exp==-2**63 else str(exp)
            lines.append(f'            Reference({k}UL,{n}UL,T(false,{ep},UInt!{bits}.fromHexString("{c:x}")),T(true,{ex},UInt!{bits}.fromHexString("{out:x}"))),')
        lines+=['        ];','        foreach (r; references) {',
                '            if (r.n > size_t.max) continue;',
                '            assert(binomialLPMF(cast(size_t)r.k,cast(size_t)r.n,r.p) == r.value);',
                '        }','    }']
    return '\n'.join(lines+['}','// END public binomial log references'])


if __name__=='__main__':
    print(render())
