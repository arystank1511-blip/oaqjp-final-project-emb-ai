"""Provide the top-level import for the emotion detection package."""

# `EmotionDetection` is the package name required by the course.
# pylint: disable=invalid-name
from .emotion_detection import emotion_detector

__all__ = ['emotion_detector']
