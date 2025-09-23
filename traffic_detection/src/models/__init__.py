"""
Models module for vehicle detection and road area detection
"""

from .person_detection.model import VehicleDetector
from .road_detection.road_detector import RoadAreaDetector

__all__ = ['VehicleDetector', 'RoadAreaDetector']
