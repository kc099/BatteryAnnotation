"""
Battery Quality Inspection Package

A hierarchical quality assessment system for battery covers using deep learning.
"""

from .model import HierarchicalQualityModel, ComponentAwareLoss
from .dataset import ComponentQualityDataset, get_training_augmentations, get_validation_augmentations
from .train import HierarchicalQualityModule, train_hierarchical_model, create_submission_package
from .inference import QualityInference, load_model_for_inference, predict_single_image

__version__ = "1.0.0"
__author__ = "Battery Quality Team"

__all__ = [
    'HierarchicalQualityModel',
    'ComponentAwareLoss', 
    'ComponentQualityDataset',
    'get_training_augmentations',
    'get_validation_augmentations',
    'HierarchicalQualityModule',
    'train_hierarchical_model',
    'create_submission_package',
    'QualityInference',
    'load_model_for_inference',
    'predict_single_image'
] 