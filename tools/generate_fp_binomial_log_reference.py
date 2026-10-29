"""Generate independent interior Fp binomial log-mass references (mpmath 1.3.0)."""
import random


def cases(bits, broad=False):
    limit = 2**64 - 1
    result = {(1, 2, 1, -1), (1, limit, 1, -64),
              (limit-1, limit, 2**bits-1, -bits)}
    for n in (3, 32, 128, 2**32+1, 10**12, limit-1, limit):
        for k in (1, n//2, n-1):
            result.add((k, n, 1, -1))
    # Near the mode, at a one-bit perturbation of p=1/2, and across
    # both deviance-series boundaries mean=x/2 and mean=3*x/2.
    for c in (2**(bits-1)-1, 2**(bits-1), 2**(bits-1)+1):
        result.add((limit//2, limit, c, -bits))
    for c in (2**(bits-3)-1, 2**(bits-3), 2**(bits-3)+1,
              3*2**(bits-3)-1, 3*2**(bits-3), 3*2**(bits-3)+1):
        result.add((32, 128, c, -bits))
    for c,e in ((1,-2000), (1,-2**63), (2**bits-1,-2**63),
                (1,-bits-320), (1,-bits-319), (2**bits-1,-bits)):
        result.add((1, limit, c, e))
        result.add((limit//2, limit, c, e))
    rng = random.Random(157+bits)
    for _ in range(40 if broad else 4):
        n = rng.randrange(2, 2**rng.randrange(2,65))
        k = rng.randrange(1,n)
        result.add((k,n,rng.randrange(1,2**bits),-bits))
    return sorted(result)


def reference(k,n,c,e,bits,precision):
    import mpmath as mp
    with mp.workdps(precision):
        p = mp.ldexp(mp.mpf(c),e)
        # Deliberately use the direct log-gamma identity with hundreds of
        # extra decimal digits, independently of the centered D algorithm.
        value = -(mp.loggamma(n+1)-mp.loggamma(k+1)-mp.loggamma(n-k+1)
                  + k*(mp.log(c)+e*mp.log(2)) + (n-k)*mp.log1p(-p))
        exponent = int(mp.floor(mp.log(value,2)))-bits+1
        scaled = mp.ldexp(value,-exponent)
        floor = int(mp.floor(scaled))
        remainder = scaled-floor
        coef = floor+(remainder>mp.mpf('0.5') or
                      (remainder==mp.mpf('0.5') and floor%2))
        if coef == 2**bits:
            coef //= 2
            exponent += 1
        return coef,exponent


def render():
    out = ['// BEGIN centered binomial references']
    for bits in (64,128,256,512):
        block_start = len(out)
        out += ['    {', f'        alias T = Fp!{bits};',
                '        struct Reference { ulong k; ulong n; T p; T value; }',
                '        enum references = [']
        for k,n,c,e in cases(bits):
            r = reference(k,n,c,e,bits,500)
            assert r == reference(k,n,c,e,bits,600)
            coef,exponent = r
            ep = 'long.min' if e == -2**63 else str(e)
            out.append(f'            Reference({k}UL,{n}UL,T(false,{ep},UInt!{bits}.fromHexString("{c:x}")),T(true,{exponent},UInt!{bits}.fromHexString("{coef:x}"))),')
        out += ['        ];', '        foreach (r; references)',
                f'            assert(BinomialLog!{bits}.mass(r.k,r.n,r.p) == r.value);', '    }']
        block = out[block_start:]
        version = 'mir_stat_test_slow_numerics' if bits >= 256 else 'mir_stat_test'
        out[block_start:] = [f'version({version})',
                              '@safe pure nothrow @nogc unittest',
                              *(line[4:] for line in block), '']
    return '\n'.join(out+['// END centered binomial references'])


if __name__ == '__main__':
    print(render())
