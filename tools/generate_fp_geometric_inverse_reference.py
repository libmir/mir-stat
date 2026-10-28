"""Generate exact and independently stabilized geometric inverse references."""
from fractions import Fraction as F
from decimal import Decimal as D, localcontext, ROUND_CEILING


def large_count(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        p = D(2) ** -64
        lp = (1-p).ln()
        target = D('0.5').ln()
        k = int((target/lp).to_integral_value(rounding=ROUND_CEILING))-1
        assert D(k)*lp > target and D(k+1)*lp <= target
        return k


def render():
    lines = ['// BEGIN inverse references', 'version(mir_stat_test)',
             '@safe pure nothrow @nogc unittest', '{',
             '    alias Status = GeometricInverseStatus;',
             '    struct Reference { uint p, q; size_t count; }',
             '    enum references = [']
    for a in range(1,8):
        for b in range(1,16):
            p, q = F(a,8), F(b,16)
            tail = 1-p
            k = 0
            while tail > 1-q:
                tail *= 1-p
                k += 1
            assert k == 0 or (1-p)**k > 1-q
            lines.append(f'        Reference({a},{b},{k}),')
    lines += ['    ];', '    static foreach (bits; [64u,128u,256u]) {{',
              '        alias T = Fp!bits;', '        foreach (c; references) {',
              '            auto p = T(c.p); p.exponent -= 3;',
              '            auto q = T(c.q); q.exponent -= 4;',
              '            const result = geometricInvCDFChecked(q,p);',
              '            assert(result.status == Status.found && result.count == c.count);',
              '            if (c.count) assert(geometricInvCDFChecked(q,p,c.count-1).status == Status.countLimit);',
              '        }', '    }}']
    for bits in (64,128,256):
        for k in range(1,bits+1):
            cdf = 1-F(3,4)**(k+1)
            c = round(cdf * 2**bits)
            q = F(c,2**bits)
            if cdf < q < 1-F(3,4)**(k+2):
                lines += ['    {', f'        alias T = Fp!{bits};',
                          f'        enum q = T(false, -{bits}, UInt!{bits}.fromHexString("{c:x}"));',
                          '        const result = geometricInvCDFChecked(q,T(0.25));',
                          f'        assert(result.status == Status.found && result.count == {k+1});',
                          '    }']
                break
    k = large_count(250)
    assert k == large_count(350)
    lines += ['    static if (size_t.sizeof == 8) {',
              '        static foreach (bits; [64u,128u,256u]) {{',
              '            alias T = Fp!bits;', '            auto p = T(1); p.exponent -= 64;',
              '            const result = geometricInvCDFChecked(T(0.5),p);',
              f'            assert(result.status == Status.found && result.count == {k}UL);',
              '        }}', '    }', '}', '// END inverse references']
    return '\n'.join(lines)


if __name__ == '__main__':
    print(render())
