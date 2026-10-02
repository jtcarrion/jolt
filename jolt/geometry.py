"""Tilt-series projection geometry in torch: 3D points to 2D tilt-image coordinates and back.

One implementation serves bundle adjustment, gap filling and fine-tuning label projection. The
conventions (image center, frames, signs) are documented in the user guide and are cross-checked
against TeamTomo.
"""
