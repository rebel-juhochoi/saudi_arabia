"""
Models module for person detection and gender classification
"""

from .person_detection.model import PersonDetector
from .gender_classification.model import GenderClassifier

__all__ = ['PersonDetector', 'GenderClassifier']
