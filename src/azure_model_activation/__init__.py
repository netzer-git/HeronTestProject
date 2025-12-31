"""Azure model activation helper."""

__all__ = ["__version__", "activate_model"]

# Keep in sync with pyproject.toml
__version__ = "0.1.0"

from .azure_openai import activate_model
