"""
Initialize the architectures module and provide a unified interface for model creation.
"""

from .vit3d import create_model
import logging

logger = logging.getLogger(__name__)

__all__ = ['create_model']