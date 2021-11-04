from invoke import task


@task(
    name='lab1',
    optional=['seed', 'a', 'm'],
    positional=[],
    help={
        'seed': 'starting seed value',
        'a': 'multiplier',
        'm': 'divisor',
    }
)
def lab1(ctx, seed=1, a=13, m=19, *args, **kwargs):
    """
    Execute lab1

    Example:
        inv lab1 -s 1 -a 6700417 -m 524287
    """
    from mod.lab1 import Lab1

    solution = Lab1(seed=seed, a=a, m=m)
    solution.execute()

@task(
    name='lab2',
    optional=[
        'equal-a', 'equal-b', 
        'gaussian-m', 'gaussian-d',
        'exponential-l',
        'gamma-n', 'gamma-l',
        'triangular-a', 'triangular-b', 'triangular-c',
        'simpson-a', 'simpson-b',
    ],
    positional=[],
    help={
        'equal-a': 'Low border for equal distribution', 
        'equal-b': 'High border for equal distribution', 
        'gaussian-m': 'Target expected value for gaussian distribution', 
        'gaussian-d': 'Targer deviation value for gaussian distribution',
        'exponential-l': 'Lambda argument for exponential distribution',
        'gamma-n': 'Eta argument for gamma distribution', 
        'gamma-l': 'Lambda argument for gamma distibution',
        'triangular-a': 'Low border for triangular distribution',
        'triangular-b': 'High border for triangular distribution', 
        'triangular-c': 'Type of triangular distribution',
        'simpson-a': 'Low border for simpson distribution',
        'simpson-b': 'High border for simpson distribution',
    },
)
def lab2(
    ctx, 
    equal_a=0.0, 
    equal_b=10.0, 
    gaussian_m=5.0,
    gaussian_d=10.0,
    exponential_l=2.0,
    gamma_n=5,
    gamma_l=2.0,
    triangular_a=0.0,
    triangular_b=10.0,
    triangular_c=True,
    simpson_a=0.0,
    simpson_b=10.0,
):
    """
    Execute lab2

    Example:
        inv lab2
    """
    from mod.lab2 import Lab2

    solution = Lab2(
        equeal_kw={
            'a': equal_a,
            'b': equal_b,
        }, 
        gaussian_kw={
            'm': gaussian_m,
            'd': gaussian_d,
        },
        exponential_kw={
            'l': exponential_l,
        },
        gamma_kw={
            'n': gamma_n,
            'l': gamma_l,
        },
        triangural_kw={
            'a': triangular_a,
            'b': triangular_b,
            'c': triangular_c,
        },
        simpson_kw={
            'a': simpson_a,
            'b': simpson_b,
        },
    )
    solution.execute()

@task(
    name='lab3',
)
def lab3(
    ctx,
    p1=0.4,
    p2=0.4,
):
    from mod.lab3 import Lab3
    solution = Lab3(p1=p1, p2=p2)
    solution.execute()
