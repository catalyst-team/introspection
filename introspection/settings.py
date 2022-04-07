from datetime import datetime
import os

import path

PROJECT_ROOT = path.Path(os.path.dirname(__file__)).joinpath("..").abspath()
DATA_ROOT = PROJECT_ROOT.joinpath("data")
LOGS_ROOT = PROJECT_ROOT.joinpath("logs")

UTCNOW = datetime.utcnow().strftime("%y%m%d.%H%M%S")  # noqa: WPS119
