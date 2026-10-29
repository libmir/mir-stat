"""Generate public Poisson LPMF references with mpmath 1.3.0."""
from generate_fp_poisson_log_reference import reference


def render():
    lines=['// BEGIN public Poisson log references','version(mir_stat_test)',
           '@safe pure nothrow @nogc unittest','{','    import mir.bignum.fixed: UInt;']
    for bits in (64,128,256,512):
        limit=2**64-1
        cases=((1,1,0),(3,5,0),(32,32,0),(10**12,10**12,0),
               (limit,limit,0),(1,1,-2**63),(limit,1,-2000),
               (1,1,66),(limit,2**bits-1,2**63-2))
        lines+=['    {',f'        alias T = Fp!{bits};',
                '        struct Reference { ulong k; T rate; T value; }',
                '        enum references = [']
        def exponent(e):
            return 'long.min' if e==-2**63 else 'long.max-1' if e==2**63-2 else str(e)
        for k,c,e in cases:
            r=reference(k,c,e,bits,500)
            assert r==reference(k,c,e,bits,600)
            coef,exp=r
            lines.append(f'            Reference({k}UL,T(false,{exponent(e)},UInt!{bits}.fromHexString("{c:x}")),T(true,{exponent(exp)},UInt!{bits}.fromHexString("{coef:x}"))),')
        lines+=['        ];','        foreach (r; references) {',
                '            if (r.k > size_t.max) continue;',
                '            assert(poissonLPMF(cast(size_t)r.k,r.rate) == r.value);',
                '        }','    }']
    return '\n'.join(lines+['}','// END public Poisson log references'])


if __name__=='__main__':
    print(render())
