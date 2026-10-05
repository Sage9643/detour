"""Logging configuration.

Plain stdlib logging with a single key=value format. Render and most hosts collect
stdout, so no log shipping infrastructure is needed.
"""

import logging

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
