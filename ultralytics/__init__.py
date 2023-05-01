# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = '8.0.75'

from ultralytics.hub import start
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils.checks import check_yolo as checks

__all__ = '__version__', 'YOLO', 'checks', 'start'  # allow simpler import
