"""Compliance tests for the streaming decoder, run on real recordings.

The tests use a model trained on dev recordings and a recording from a CV fold.
"""
import glob
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.io import FS, list_recordings, read_bdf
from src.inference.decoder import WIN, StreamingDecoder
from src.preprocessing.features import extract_v2
from src.preprocessing.quality import invalid_mask

MODEL = os.environ.get("BCI_MODEL_DIR", "checkpoints/dev_D2")
ROOT = "EEG_dataset"


@pytest.fixture(scope="module")
def signal():
    bdf, _ = [p for p in list_recordings(ROOT) if os.path.basename(os.path.dirname(p[0])) == "040"][0]
    x, _, _ = read_bdf(bdf)
    return x[60 * FS : 60 * FS + 150 * FS].astype(np.float64)  # 2.5 min, stays fast


def run(x, chunk, **kw):
    d = StreamingDecoder(MODEL, **kw)
    out = []
    for i in range(0, len(x), chunk):
        out += d.push(x[i : i + chunk])
    return out


def test_accepts_chunks_and_returns_probabilities(signal):
    out = run(signal, 125)
    assert len(out) == len(signal) // FS
    assert all(0.0 <= d.p_task <= 1.0 for d in out)
    assert all(d.state == int(d.p_task >= 0.5) for d in out)
    assert [d.sample_index for d in out] == list(range(FS - 1, len(signal), FS))


@pytest.mark.parametrize("chunk", [1, 7, 125, 250, 333, 1500, 5000])
def test_chunk_size_does_not_change_output(signal, chunk):
    x = signal[: 40 * FS]
    ref = [d.p_task for d in run(x, 250)]
    got = [d.p_task for d in run(x, chunk)]
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("adapt", [True, False])
def test_future_perturbation_leaves_past_decisions_unchanged(signal, adapt):
    cut = 100 * FS
    a = run(signal, 250, adapt=adapt)
    fut = signal.copy()
    fut[cut:] = np.random.default_rng(0).normal(0, 500, len(fut) - cut)
    b = run(fut, 250, adapt=adapt)
    past_a = [d.p_task for d in a if d.sample_index < cut]
    past_b = [d.p_task for d in b if d.sample_index < cut]
    assert len(past_a) == 100
    assert np.array_equal(past_a, past_b)
    # The perturbation must actually reach decisions after the cut.
    assert not np.allclose([d.p_task for d in a if d.sample_index >= cut], [d.p_task for d in b if d.sample_index >= cut])


def test_receptive_field_is_six_seconds_without_adaptation(signal):
    """With adaptation off, samples older than 1500 back have no effect at all."""
    t = 120 * FS
    x = signal.copy()
    x[: t - WIN] = np.random.default_rng(1).normal(0, 300, t - WIN)
    a = {d.sample_index: d.p_task for d in run(signal, 250, adapt=False)}
    b = {d.sample_index: d.p_task for d in run(x, 250, adapt=False)}
    for k in range(t - 1, len(signal), FS):
        assert a[k] == b[k]
    # And the newest sample of the window does matter.
    y = signal.copy()
    y[t - WIN : t] = np.random.default_rng(2).normal(0, 300, WIN)
    c = {d.sample_index: d.p_task for d in run(y, 250, adapt=False)}
    assert a[t - 1] != c[t - 1]


def test_window_passed_to_model_never_exceeds_six_seconds(signal):
    assert max(d.window_samples for d in run(signal, 250)) == WIN


def test_block_and_sample_by_sample_reference_agree(signal):
    x = signal[: 20 * FS]
    d = StreamingDecoder(MODEL)
    ref = []
    for v in x:  # one sample at a time, the causal reference
        ref += d.push(np.array([v]))
    block = run(x, 500)
    assert [r.p_task for r in ref] == [b.p_task for b in block]


def test_online_history_grows_only_from_past_full_windows(signal):
    d = StreamingDecoder(MODEL)
    d.push(signal[: 30 * FS])
    assert d.rank.n == 30 - 6 + 1


def test_missing_data_handled(signal):
    x = signal[: 60 * FS].copy()
    x[20 * FS : 30 * FS] = 0.024  # the zero-fill value decoded from the BDF
    x[40 * FS : 42 * FS] = np.nan
    out = run(x, 250)
    by = {d.sample_index: d for d in out}
    inside = by[25 * FS - 1]
    assert not inside.evaluated and inside.reason == "sample_invalid"
    assert inside.p_task == by[20 * FS - 1].p_task
    # Too little clean signal right after the gap: score still held.
    assert not by[31 * FS - 1].evaluated
    # Usable clean suffix: evaluated on the suffix only, no splice across the gap.
    after = by[35 * FS - 1]
    assert after.evaluated and after.window_samples == 5 * FS
    assert after.reason == "gap_in_window"
    assert not by[41 * FS - 1].evaluated
    assert np.isfinite([d.p_task for d in out]).all()


def test_gap_splice_is_not_used(signal):
    """Changing samples before a gap must not alter a decision whose window holds the gap."""
    x = signal[: 40 * FS].copy()
    x[20 * FS : 22 * FS] = np.nan
    y = x.copy()
    y[17 * FS : 20 * FS] = np.random.default_rng(3).normal(0, 200, 3 * FS)
    a = {d.sample_index: d.p_task for d in run(x, 250, adapt=False)}
    b = {d.sample_index: d.p_task for d in run(y, 250, adapt=False)}
    assert a[26 * FS - 1] == b[26 * FS - 1]


def test_rail_saturation_detected():
    x = np.zeros(10) + 50.0
    x[3:6] = 403333.0
    m = invalid_mask(x)
    assert m[3:6].all() and not m[:3].any() and not m[6:].any()


def test_reset_gives_recording_isolation(signal):
    x = signal[: 30 * FS]
    d = StreamingDecoder(MODEL)
    first = [o.p_task for o in d.push(x)]
    d.push(signal[30 * FS : 60 * FS])
    d.reset()
    again = [o.p_task for o in d.push(x)]
    assert first == again


def test_warmup_outputs_before_six_seconds(signal):
    out = run(signal[: 10 * FS], 250)
    assert out[0].reason.endswith("too_short") and out[0].p_task == 0.5
    assert out[2].evaluated and out[2].window_samples == 3 * FS and out[2].reason == "warmup"


def test_features_use_only_the_given_window(signal):
    w = signal[:WIN][None, :]
    f1 = extract_v2(w)
    f2 = extract_v2(signal[:WIN].copy()[None, :])
    assert np.array_equal(f1, f2)


def test_auc_ready_and_realtime_on_cpu(signal):
    t0 = time.perf_counter()
    out = run(signal, 250)
    cost = (time.perf_counter() - t0) / len(out)
    assert np.std([d.p_task for d in out if d.evaluated]) > 0
    assert cost < 1.0, f"{cost:.3f}s per 1-s decision"


@pytest.mark.parametrize("chunk", [100, 250, 333, 1000])
def test_process_chunk_returns_one_result_per_call_and_keeps_history_stable(signal, chunk):
    x = signal[: 40 * FS]
    d = StreamingDecoder(MODEL)
    results = [d.process_chunk(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    assert len(results) == int(np.ceil(len(x) / chunk))
    assert [r.sample_index for r in results] == [min(i + chunk, len(x)) - 1 for i in range(0, len(x), chunk)]
    ref = StreamingDecoder(MODEL)
    grid = {o.sample_index: o.p_task for o in ref.push(x)}
    for r in results:
        if r.sample_index in grid:
            assert r.p_task == grid[r.sample_index]
    assert d.rank.n == ref.rank.n


def test_process_chunk_is_causal(signal):
    x = signal[: 30 * FS]
    y = signal[: 40 * FS].copy()
    y[30 * FS :] = np.random.default_rng(5).normal(0, 400, len(y) - 30 * FS)
    a, b = StreamingDecoder(MODEL), StreamingDecoder(MODEL)
    ra = [a.process_chunk(x[i : i + 333]) for i in range(0, len(x), 333)]
    rb = [b.process_chunk(y[i : i + 333]) for i in range(0, len(y), 333)]
    for r, s in zip(ra[:-1], rb):
        assert r.p_task == s.p_task


def test_multi_window_features_stay_inside_the_window(signal):
    from src.preprocessing.features import extract_multi

    w = signal[: 2 * WIN][None, -WIN:]
    f = extract_multi(w, (1500, 500))[0]
    assert np.array_equal(f[: len(f) // 2], extract_v2(w)[0])
    assert np.array_equal(f[len(f) // 2 :], extract_v2(w[:, -500:])[0])
    # Samples older than the 6-s window cannot enter: the input is the window itself.
    assert extract_multi(signal[None, WIN : 2 * WIN], (1500, 500)).shape == (1, len(f))
