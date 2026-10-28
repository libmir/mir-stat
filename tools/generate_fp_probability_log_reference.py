"""Generate independently stabilized Fp Bernoulli/geometric log references."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import random


def rounded(x, bits):
    x=F(x)
    e=x.numerator.bit_length()-x.denominator.bit_length()
    if x < F(2)**e: e-=1
    e-=bits-1
    c=round(x/F(2)**e)
    if c==2**bits: c//=2; e+=1
    return c,e


def reference(c,e,k,bits,precision):
    with localcontext() as ctx:
        ctx.prec=precision
        lp=-(D(c).ln()+D(e)*D(2).ln())
        if e < -10000:
            # The relative correction in -log(1-p)/p is at most
            # p/(2*(1-p)), hence below p for these tiny inputs.
            # Here it is vastly below half an output ulp. k*p is likewise
            # negligible beside -log(p), for every 64-bit count.
            lc=(c,e)
            return rounded(lp,bits),lc,rounded(lp,bits)
        p=D(c)*D(2)**e
        lq=-(1-p).ln()
        return rounded(lp,bits),rounded(lq,bits),rounded(lp+D(k)*lq,bits)


def cases(bits, broad=False):
    cs={(1,-1),(3,-2),(1,-bits),(2**bits-1,-bits),
        (2**(bits-1)-1,-bits),(2**(bits-1)+1,-bits),
        (1,-2000),(1,-2*bits-256),(1,-2*bits-257),
        (2**bits-1,-2*bits-256),(2**bits-1,-2*bits-257)}
    rng=random.Random(157+bits)
    for _ in range(120 if broad else 8):
        cs.add((rng.randrange(1,2**bits),-rng.randrange(bits,bits+300)))
    # Raw subnormal representations at the Fp exponent floor.
    cs.update((c,-2**63) for c in (1,3,2**(bits-1),2**bits-1))
    return [(c,e,k) for c,e in sorted(cs) for k in ((0,1,17,2**64-1) if broad else (17,2**64-1))]


def render(broad=False, target="helper"):
    out=[f'// BEGIN {target} probability log references','version(mir_stat_test)',
         '@safe pure nothrow @nogc unittest','{',
         '    import mir.bignum.fixed: UInt;']
    for bits in (64,128,256):
        out += ['    {',f'        alias T = Fp!{bits};',
                '        struct Reference { T p; '+('ulong k; T logMass;' if target=='geometric' else 'T logP; T logQ;')+' }',
                '        enum references = [']
        def value(c,e,sign=True):
            exp='long.min' if e==-2**63 else str(e)
            return f'T({str(sign).lower()},{exp},UInt!{bits}.fromHexString("{c:x}"))'
        if target == 'helper':
            out.insert(len(out)-3, f'        enum bits = {bits};')
        seen = set()
        for c,e,k in cases(bits,broad):
            r=reference(c,e,k,bits,900)
            assert r==reference(c,e,k,bits,1100)
            fields = ([value(c,e,False),f'{k}UL',value(*r[2])] if target=='geometric'
                      else [value(c,e,False)]+[value(*v,sign=target!='helper') for v in r[:2]])
            line = '            Reference('+','.join(fields)+'),'
            if line not in seen:
                out.append(line)
                seen.add(line)
        out += ['        ];','        foreach (r; references) {']
        if target == 'helper':
            out += ['            assert(ProbabilityLog!bits.probability(r.p).finish!bits == r.logP);',
                    '            assert(ProbabilityLog!bits.complement(r.p).finish!bits == r.logQ);']
        elif target == 'bernoulli':
            out += ['            assert(bernoulliLPMF(true,r.p) == r.logP);',
                    '            assert(bernoulliLPMF(false,r.p) == r.logQ);']
        else:
            out += ['            if (r.k <= size_t.max)',
                    '                assert(geometricLPMF(cast(size_t)r.k,r.p) == r.logMass);']
        out += ['        }','    }']
    return '\n'.join(out+['}',f'// END {target} probability log references'])



if __name__=='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broad', action='store_true')
    parser.add_argument('--target', choices=('helper','bernoulli','geometric'), default='helper')
    args = parser.parse_args()
    print(render(args.broad,args.target))
