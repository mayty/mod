from structlog import get_logger
from typing import Dict, List, Any
from random import random, randrange
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from math import log, prod

from mod.common.processor import BaseProcessor
from mod.common.statistics import show_properties

logger = get_logger(__name__)


class BaseGenerator:
    distribution_type = 'error'

    def transform(self, value: float) -> float:
        raise NotImplementedError()

    def transform_bulk(self, values: List[float]) -> List[float]:
        return [self.transform(value) for value in values]


class EqualDistributionGenerator(BaseGenerator):
    distribution_type = 'equal distribution'

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        assert a < b, 'a must be less then b'
        self.a = a
        self.b = b

    def transform(self, value: float) -> float:
        return self.a + (self.b - self.a) * value


class GaussianDistributionGenerator(BaseGenerator):
    distribution_type = 'gaussian distibution'
    def __init__(self, m: float, d: float) -> None:
        super().__init__()
        self.m = m
        self.d = d

    def transform_bulk(self, values: List[float]) -> List[float]:
        result = []
        for i in range(len(values)):
            s = sum(values[randrange(0, len(values) - 1)] for j in range(6))
            result.append(self.m + self.d * (2 ** 0.5) * (s - 3))
        return result


class ExponentialDistributionGenerator(BaseGenerator):
    distribution_type = 'exponential distribution'

    def __init__(self, l: float) -> None:
        super().__init__()
        self.l = l

    def transform(self, value: float) -> float:
        return - log(value) / self.l


class GammaDistributionGenerator(BaseGenerator):
    distribution_type = 'gamma distribution'

    def __init__(self, l: float, n: int) -> None:
        super().__init__()
        self.l = l
        self.n = n

    def transform_bulk(self, values: List[float]) -> List[float]:
        result = []
        for i in range(len(values)):
            p = prod(values[randrange(0, len(values) - 1)] for j in range(self.n))
            result.append(-log(p)/self.l)
        return result


class TrianguralDistributionGenerator(BaseGenerator):
    distribution_type = 'triangural distribution'

    def __init__(self, a: float, b: float, c: bool) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def transform_bulk(self, values: List[float]) -> List[float]:
        result = []
        for i in range(len(values)):
            if self.c:
                p = min(values[randrange(0, len(values) - 1)] for j in range(2))
            else:
                p = max(values[randrange(0, len(values) - 1)] for j in range(2))
            result.append(self.a + p * (self.b - self.a))
        return result


class SimpsonDistributionGenerator(BaseGenerator):
    distribution_type = 'simpson distribution'

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
        self.equal_generator = EqualDistributionGenerator(a=a/2, b=b/2)

    def transform_bulk(self, values: List[float]) -> List[float]:
        equal_values = self.equal_generator.transform_bulk(values)
        result = []
        for i in range(len(values)):
            result.append(sum(equal_values[randrange(0, len(equal_values) - 1)] for j in range(2)))
        return result


class Lab2(BaseProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.numbers = [random() for i in range(100000)]
        self.equal_kw = kwargs.get('equeal_kw') or {}
        self.gaussian_kw = kwargs.get('gaussian_kw') or {}
        self.exponential_kw = kwargs.get('exponential_kw') or {}
        self.gamma_kw = kwargs.get('gamma_kw') or {}
        self.triangural_kw = kwargs.get('triangural_kw') or {}
        self.simpson_kw = kwargs.get('simpson_kw') or {}

    def show_properties(self, generator: BaseGenerator, ax: Axes) -> None:
        logger.info('Generating dataset', type=generator.distribution_type)
        values = generator.transform_bulk(self.numbers)
        show_properties(values)
        ax.hist(values, bins=20)  # type: ignore
        ax.set_title(generator.distribution_type)

    def execute(self) -> None:
        fig, axs = plt.subplots(  # type: ignore
            2,3,
            gridspec_kw={
                'hspace': 0.29,
                'left': 0.05,
                'right': 0.95,
            }
        )
        for ax, generator in zip(
            axs.flat,  # type: ignore
            (
                EqualDistributionGenerator(**self.equal_kw),
                GaussianDistributionGenerator(**self.gaussian_kw),
                ExponentialDistributionGenerator(**self.exponential_kw),
                GammaDistributionGenerator(**self.gamma_kw),
                SimpsonDistributionGenerator(**self.simpson_kw),
                TrianguralDistributionGenerator(**self.triangural_kw),
            )
        ):
            ax.yaxis.set_ticklabels([])
            self.show_properties(generator, ax)
        plt.show()