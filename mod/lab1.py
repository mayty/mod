from collections import defaultdict
from math import floor
from structlog import get_logger
from typing import Dict, List
import matplotlib.pyplot as plt

from mod.common.processor import BaseProcessor

logger = get_logger(__name__)


class Lab1(BaseProcessor):
    def __init__(self, seed: int, a: int, m: int) -> None:
        super().__init__()
        assert a > 0, "a must be positive"
        assert m > 0, "m must be positive"
        assert seed > 0, "seed must be positive"
        self._starting_seed = seed
        self._seed = self._starting_seed
        self._a = a
        self._m = m
        self._curr_index = 0

    def generate_next(self) -> float:
        mul = self._seed * self._a
        quotient = floor(mul / self._m)
        remainder = mul - (quotient * self._m)
        self._seed = remainder
        self._curr_index += 1
        return remainder / self._m

    def reset(self) -> None:
        self._seed = self._starting_seed
        self._curr_index = 0

    def execute(self) -> None:
        x_v = None
        values_occurrence: Dict[float, List[int]] = defaultdict(list)
        numbers = []
        for i in range((10 ** 6) // 5):
            x_v = self.generate_next()
            values_occurrence[x_v].append(self._curr_index)
            numbers.append(x_v)
        assert x_v is not None
        while len(values_occurrence[x_v]) < 2:
            current_value = self.generate_next()
            values_occurrence[current_value].append(self._curr_index)
            numbers.append(current_value)

        i1, i2 = values_occurrence[x_v][0:2]
        P = i2 - i1

        found_match = False
        i3 = -1
        while not found_match:
            i3 += 1
            while len(numbers) <= (i3 + P):
                current_value = self.generate_next()
                numbers.append(current_value)
                values_occurrence[current_value].append(self._curr_index)
            if numbers[i3] == numbers[i3 + P]:
                found_match = True

        if len(numbers) % 2 != 0:
            current_value = self.generate_next()
            numbers.append(current_value)
            values_occurrence[current_value].append(self._curr_index)

        K = 0
        for i in range(len(numbers) // 2):
            if (numbers[2 * i] ** 2) + (numbers[(2 * i) + 1] ** 2) < 1:
                K += 1

        probability = 2 * K / len(numbers)

        L = i3 + P
        avg = (
            sum(len(ids) * value for value, ids in values_occurrence.items())
            / self._curr_index
        )
        dispersion = sum(
            len(ids) * ((key - avg) ** 2) for key, ids in values_occurrence.items()
        ) / (self._curr_index - 1)
        standard_deviation = dispersion ** 0.5
        logger.info("Samples info", sample_size=self._curr_index)
        logger.info(
            "Task 2",
            average=avg,
            dispersion=dispersion,
            standard_deviation=standard_deviation,
        )
        logger.info("Task 3", probability=probability)
        logger.info("Task 4", period_size=P, aperiodic_size=L)

        plt.hist(numbers, bins=20)  # type: ignore
        plt.show()
