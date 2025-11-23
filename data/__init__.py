"""
Data package for FSS Network
"""

from .dataset import SegPPD101Dataset, build_dataloader

__all__ = ['SegPPD101Dataset', 'build_dataloader']
