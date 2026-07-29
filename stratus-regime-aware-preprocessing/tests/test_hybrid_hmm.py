import numpy as np
import pandas as pd

from stratus.hybrid_hmm import SelectiveHybridHMM


class DummyClassifier:
    classes_ = np.array([0, 1, 4, 5])

    def predict_proba(self, features):
        # Alternating local evidence; valid probabilities in class order.
        out = np.tile(np.array([0.55, 0.25, 0.15, 0.05]), (len(features), 1))
        if len(out) > 1:
            out[1::2] = np.array([0.20, 0.60, 0.15, 0.05])
        return out


def diagnostics(missing, run_lengths):
    n = len(missing)
    return pd.DataFrame({
        "m_t": np.asarray(missing, dtype=int),
        "r_t": np.asarray(run_lengths, dtype=int),
        "c_t": np.linspace(0, 5, n),
        "alpha_t": np.linspace(0, 2, n),
        "u_t": np.linspace(0, 1, n),
        "p_t": np.zeros(n),
    })


def model():
    transition = np.full((4, 4), np.log(0.05))
    np.fill_diagonal(transition, np.log(0.85))
    return SelectiveHybridHMM(
        classifier=DummyClassifier(),
        transition_log=transition,
        start_log=np.log(np.full(4, 0.25)),
        c_value=1.0,
    )


def test_complete_run_duration_hard_constrains_loss_states():
    # At 4 Hz, 0.5 seconds equals 2 samples.
    frame = diagnostics(
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 2, 2, 0, 4, 4, 4, 4, 0],
    )
    pred = model().predict(frame, fs_hz=4.0, transition_strength=1.0)
    assert np.all(pred[1:3] == 2)  # LossShort
    assert np.all(pred[4:8] == 3)  # LossLong


def test_transition_strength_zero_is_exact_pointwise_control():
    frame = diagnostics([0, 0, 0, 0], [0, 0, 0, 0])
    m = model()
    emissions = m.emission_log_probabilities(frame, fs_hz=60.0)
    expected = np.array([0, 1, 0, 1])
    assert np.array_equal(np.argmax(emissions, axis=1), expected)
    assert np.array_equal(m.predict(frame, fs_hz=60.0, transition_strength=0.0), expected)


def test_missing_samples_split_finite_decoding_blocks():
    frame = diagnostics([0, 0, 1, 1, 0, 0], [0, 0, 2, 2, 0, 0])
    pred = model().predict(frame, fs_hz=4.0, transition_strength=1.0)
    assert np.all(pred[2:4] == 2)
    assert pred[0] in {0, 1, 4, 5}
    assert pred[4] in {0, 1, 4, 5}
