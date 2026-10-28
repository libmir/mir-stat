from decimal import Decimal as D, localcontext, ROUND_HALF_EVEN
from fractions import Fraction

def reference(n,bits,precision):
    with localcontext() as ctx:
        ctx.prec=precision
        x=D(n).ln()
        f=Fraction(x)
        exponent=f.numerator.bit_length()-f.denominator.bit_length()
        if f < Fraction(2)**exponent: exponent-=1
        exponent-=bits-1
        coefficient=int((x/(D(2)**exponent)).to_integral_value(rounding=ROUND_HALF_EVEN))
        return coefficient,exponent

def render():
    counts=sorted(set(list(range(2,18))+[2**k+d for k in (16,32,53,63,64) for d in (-1,0,1) if 2<=2**k+d<=2**64]))
    lines=['// BEGIN discrete-uniform log references','version(mir_stat_test)','@safe pure nothrow @nogc unittest','{']
    for bits in (64,128,256):
        lines+=['    {',f'        alias T = Fp!{bits};','        struct Reference { ulong upper; T logMass; }','        enum references = [']
        for n in counts:
            c,e=reference(n,bits,500)
            assert (c,e)==reference(n,bits,600)
            lines.append(f'            Reference({n-1}UL,T(true,{e},UInt!{bits}.fromHexString("{c:x}"))),')
        lines+=['        ];','        foreach (c; references) {','            if (c.upper > size_t.max) continue;','            const upper = cast(size_t)c.upper;','            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);','            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);','        }','    }']
    lines+=['}','// END discrete-uniform log references']
    return '\n'.join(lines)

if __name__=='__main__':
    print(render())
