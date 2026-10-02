import numpy as np
import torch
from scipy.spatial import cKDTree

from jolt.mlesac import MlesacSettings, run_mlesac
from tests.synthetic import synthetic_picks


def test_mlesac_recovers_synthetic_particles():
    pts, picks, _ = synthetic_picks(n_particles=25)
    result = run_mlesac(picks, (256, 256), MlesacSettings(), device=torch.device("cpu"), log=lambda m: None)
    d, _ = cKDTree(result.X_3d).query(pts)
    assert np.all(d < 1.0)
    assert len(result.X_3d) == 25
