import numpy as np

from hybrid import combine_probs, fit_alpha


def test_combine_probs_is_normalized() -> None:
    ml = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    phys = np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float64)
    out = combine_probs(ml, phys, 0.25)
    assert out.shape == ml.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out >= 0.0)


def test_fit_alpha_prefers_ml_when_it_is_better() -> None:
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    ml = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
        dtype=np.float64,
    )
    phys = np.array(
        [[0.4, 0.6], [0.6, 0.4], [0.5, 0.5], [0.55, 0.45]],
        dtype=np.float64,
    )
    alpha = fit_alpha(ml, phys, y, grid=np.linspace(0.0, 1.0, 11))
    assert 0.5 <= alpha <= 1.0
