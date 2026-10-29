"""Generate guarded Poisson log-mass references with mpmath 1.3.0."""
import random


def cases(bits):
    limit=2**64-1
    values=set()
    for k in (1,2,3,32,1000,10**12,limit):
        shift=bits-k.bit_length()
        coefficient=k<<shift
        for offset in (-1,0,1):
            if 0<coefficient+offset<2**bits:
                values.add((k,coefficient+offset,-shift))
        values.add((k,k,0)) # Unnormalized representation of the exact mean.
    for mean in (16,48):
        shift=bits-mean.bit_length()
        for offset in (-1,0,1):
            values.add((32,(mean<<shift)+offset,-shift))
    for k in (1,limit):
        for c,e in ((1,-2000),(1,-2**63),(2**bits-1,-2**63),
                    (1,66),(1,2**63-2),(2**bits-1,2**63-2)):
            values.add((k,c,e))
        for offset in (-1,0,1):
            values.add((k,2**(bits-1)+offset,193))
            # Rates around one fixed-point grid unit exercise right shifts
            # that retain a unit or truncate the rate to zero.
            values.add((k,2**(bits-1)+offset,-(2*bits+319)))
    rng=random.Random(15700+bits)
    for _ in range(8):
        values.add((rng.randrange(1,2**64),rng.randrange(1,2**bits),rng.randrange(-100,201)-bits))
    return sorted(values)


def reference(k,c,e,bits,precision):
    import mpmath as mp
    with mp.workdps(precision):
        rate=mp.ldexp(mp.mpf(c),e)
        value=rate+mp.loggamma(k+1)-k*(mp.log(c)+e*mp.log(2))
        exponent=int(mp.floor(mp.log(value,2)))-bits+1
        scaled=mp.ldexp(value,-exponent)
        low=int(mp.floor(scaled))
        remainder=scaled-low
        coefficient=low+(remainder>mp.mpf('0.5') or
                         (remainder==mp.mpf('0.5') and low%2))
        if coefficient==2**bits:
            coefficient//=2
            exponent+=1
        return coefficient,exponent


def render():
    lines=['// BEGIN Poisson log references','version(mir_stat_test)',
           '@safe pure nothrow @nogc unittest','{']
    for bits in (64,128,256,512):
        block_start = len(lines)
        lines+=['    {',f'        alias T = Fp!{bits};',
                '        struct Reference { ulong k; T rate; T value; }',
                '        enum references = [']
        def exponent(e):
            return 'long.min' if e==-2**63 else 'long.max-1' if e==2**63-2 else str(e)
        for k,c,e in cases(bits):
            r=reference(k,c,e,bits,500)
            assert r==reference(k,c,e,bits,600)
            coef,exp=r
            lines.append(f'            Reference({k}UL,T(false,{exponent(e)},UInt!{bits}.fromHexString("{c:x}")),T(true,{exponent(exp)},UInt!{bits}.fromHexString("{coef:x}"))),')
        lines+=['        ];','        foreach (r; references)',
                f'            assert(PoissonLog!{bits}.mass(r.k,r.rate) == r.value);','    }']
        if bits >= 512:
            block = lines[block_start:]
            lines[block_start:] = [
                '    // Keep the expensive wide reference sweep in the slow suite.',
                '    version(mir_stat_test_slow_numerics)', '    {',
                *('    ' + line for line in block), '    }']
    return '\n'.join(lines+['}','// END Poisson log references'])


if __name__=='__main__':
    print(render())
