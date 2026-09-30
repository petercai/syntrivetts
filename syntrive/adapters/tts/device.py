import logging

logger = logging.getLogger(__name__)


def detect_best_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        logger.debug("detect_best_device: torch not installed, defaulting to cpu")
        return "cpu"
