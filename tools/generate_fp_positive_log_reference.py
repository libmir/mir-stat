"""Generate guarded positive-value and exact-count logarithm references."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import random


def reference(c,e,bits,precision):
    with localcontext() as ctx:
        ctx.prec=precision
        value=D(c).ln()+D(e)*D(2).ln()
    if not value: return False,0,0
    negative=value<0
    x=abs(F(value))
    exponent=x.numerator.bit_length()-x.denominator.bit_length()
    if x < F(2)**exponent: exponent-=1
    exponent-=bits-1
    coefficient=round(x/F(2)**exponent)
    if coefficient==2**bits: coefficient//=2; exponent+=1
    return negative,exponent,coefficient


def render():
    out=['// BEGIN positive logarithm references','version(mir_stat_test)',
         '@safe pure nothrow @nogc unittest','{']
    for bits in (64,128,256,512):
        width=bits+192
        rng=random.Random(157+bits)
        cases={(1,0),(1,-1),(1,1),(3,-1),(3,-2),(1,-2**63),
               (1,2**63-2),(2**bits-1,-bits),(2**(bits-1)+1,1-bits),
               (2**bits-1,-2**63),(2**bits-1,2**63-2)}
        cases.update((rng.randrange(1,2**bits),rng.randrange(-2000,2001)) for _ in range(8))
        counts=sorted({1,2,3,17,2**32-1,2**32+1,2**64-1,2**64,2**64+1,2**128-1})
        def expected(c,e):
            r=reference(c,e,width,500)
            assert r==reference(c,e,width,600)
            sign,exponent,coef=r
            return f'W({str(sign).lower()},{exponent},UInt!{width}.fromHexString("{coef:x}"))' if coef else 'W(0)'
        out += ['    {',f'        alias L = ProbabilityLog!{bits};',f'        alias W = Fp!{width};',
                f'        alias T = Fp!{bits};','        struct ValueReference { T value; W logarithm; }',
                '        enum values = [']
        for c,e in sorted(cases):
            exp='long.min' if e==-2**63 else 'long.max-1' if e==2**63-2 else str(e)
            out += [f'            ValueReference(T(false,{exp},UInt!{bits}.fromHexString("{c:x}")),{expected(c,e)}),']
        out += ['        ];','        foreach (r; values)',
                '            assert(L.logPositive(r.value) == r.logarithm);',
                '        struct CountReference { UInt!128 count; W logarithm; }','        enum counts = [']
        for c in counts:
            out += [f'            CountReference(UInt!128.fromHexString("{c:x}"),{expected(c,0)}),']
        out += ['        ];','        foreach (r; counts)',
                '            assert(L.logCount(r.count) == r.logarithm);','    }']
    return '\n'.join(out+['}','// END positive logarithm references'])


if __name__=='__main__':
    print(render())
