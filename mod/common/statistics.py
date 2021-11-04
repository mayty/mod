from structlog import get_logger
from typing import List

logger = get_logger(__name__)


def show_properties(values: List[float]) -> None:
    avg = sum(values) / len(values)
    dispersion = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = dispersion ** 0.5
    logger.info("Samples info", sample_size=len(values))
    logger.info(
        "Properties",
        average=avg,
        dispersion=dispersion,
        standard_deviation=standard_deviation,
    )
