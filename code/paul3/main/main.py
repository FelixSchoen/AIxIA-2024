import sys
from pathlib import Path

import torch


def main():
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

    # Set up logging
    from paul3.utils.paul_logging import get_logger
    LOGGER = get_logger(__name__)
    LOGGER.info(
        f"Running PyTorch {torch.__version__} with CUDA {torch.version.cuda}")

    # Load settings
    from paul3.utils.settings import Settings
    _ = Settings()

    # Override representation
    override_repr = True

    if override_repr:
        normal_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = lambda self: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"

    # Handle arguments
    from paul3.main.argument_handling import _parse_arguments, _handle_arguments
    args = _parse_arguments()
    _handle_arguments(args)


# TODO Weight averaging?
# TODO Refactor num_workers to config file
if __name__ == '__main__':
    main()
