from invoke import task


@task(
    name='lab1',
    optional=('seed', 'a', 'm'),
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
    """
    from mod.lab1 import Lab1

    solution = Lab1(seed=seed, a=a, m=m)
    solution.execute()
