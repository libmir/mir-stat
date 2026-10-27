"""Generate geometric PMF references at two independent Decimal precisions."""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_HALF_EVEN

def generate(precision):
    rows=[]
    with localcontext() as ctx:
        ctx.prec=precision
        ln2=D(2).ln()
        for bits in (64,128,256):
            for k,a,e in [(1,1,-1),(17,3,-2),(1000,1,-3),(2**64-1,1,-bits-1),(2**64-1,1,-64),(2**60,1,-60),(7,2**bits-1,-bits),(0,1,-1100),(1,1,-1100),(1000,1,-1100),(65,1,-bits),(100,1,-10)]:
                p=D(a)*D(2)**e
                logp=p.ln()+D(k)*(1-p).ln()
                exponent=int((logp/ln2).to_integral_value(rounding=ROUND_FLOOR))-bits+1
                coefficient=int((logp-exponent*ln2).exp().to_integral_value(rounding=ROUND_HALF_EVEN))
                if coefficient==1<<bits:
                    coefficient>>=1; exponent+=1
                rows.append((bits,k,a,e,coefficient,exponent))
    return rows

def render(rows):
    lines=['// BEGIN geometric references','version(mir_stat_test)', '@safe pure nothrow @nogc unittest', '{','    import mir.bignum.fixed: UInt;', '    import mir.stat.distribution.negative_binomial: negativeBinomialPMF;']
    for bits in (64,128,256):
        lines += ['    {',f'        alias T = Fp!{bits};','        struct Reference { ulong k; T p, expected; }','        enum references = [']
        for b,k,a,e,c,x in rows:
            if b==bits:
                lines.append(f'            Reference({k}UL, T(false, {e-bits+a.bit_length()}, UInt!{bits}.fromHexString("{a << (bits-a.bit_length()):x}")), T(false, {x}, UInt!{bits}.fromHexString("{c:x}"))),')
        lines += ['        ];','        foreach (c; references) {','            if (c.k > size_t.max) continue;','            const actual = geometricPMF(cast(size_t)c.k, c.p);','            assert(actual == c.expected);','            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);','        }','    }']
    return '\n'.join(lines+['}', '// END geometric references'])

if __name__=='__main__':
    rows=generate(500)
    assert rows==generate(600)
    print(render(rows))
