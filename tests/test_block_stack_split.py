"""The block-stack train/val split must put the LONG trajectories in val and hit ~90/10 by frames.

This is the rule a random seed-0 draw cannot express, and getting it wrong is not loud: the
dataset builds fine, trains fine, and only the open-loop rollout horizon quietly collapses to
the length of the shortest val episode. That is exactly what happened on lego. So it gets a test.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from quickdraw.data.processors import _longest_first_val   # noqa: E402


def test_val_holds_the_longest_episodes():
    lens = [100, 4000, 200, 150, 3000, 120, 90, 300, 110, 130]
    val = _longest_first_val(lens, 0.1)
    assert val == {1, 4}, val                    # the two longest; the floor of 2 binds here


def test_hits_the_fraction_by_frames_not_by_count():
    lens = [100] * 100                           # all equal -> 10 episodes for 10% of frames
    val = _longest_first_val(lens, 0.1)
    assert sum(lens[i] for i in val) / sum(lens) >= 0.1
    assert len(val) == 10, len(val)


def test_takes_strictly_in_descending_length_order():
    lens = [5, 50, 5, 40, 5, 30, 5, 20, 5, 10, 5, 60, 5, 70]
    val = _longest_first_val(lens, 0.5)
    chosen = sorted((lens[i] for i in val), reverse=True)
    assert chosen == sorted(lens, reverse=True)[:len(chosen)], chosen
    rest = [lens[i] for i in range(len(lens)) if i not in val]
    assert min(chosen) >= max(rest), "a shorter episode was taken over a longer one"


def test_shortest_val_episode_is_long_which_is_the_whole_point():
    """The evaluable rollout horizon is the SHORTEST val episode, so it must beat the median."""
    lens = [50, 60, 55, 900, 70, 800, 65, 58, 62, 51] * 5
    val = _longest_first_val(lens, 0.1)
    import statistics
    assert min(lens[i] for i in val) > statistics.median(lens)


def test_lands_closest_to_the_target_rather_than_overshooting():
    """Accumulate-until-crossed turns a 10% request into 27% here. Closest-prefix gives 18%."""
    lens = [1000, 800] + [100] * 60
    val = _longest_first_val(lens, 0.1)
    got = sum(lens[i] for i in val) / sum(lens)
    assert val == {0, 1}, val
    assert abs(got - 0.10) < abs(0.27 - 0.10), f"{got:.3f} is worse than overshooting"


def test_val_is_never_a_single_episode():
    """One val episode means one session's lighting and layout ARE the validation signal."""
    lens = [10000] + [100] * 40
    assert len(_longest_first_val(lens, 0.1)) >= 2


def test_never_empties_train_or_val():
    for lens in ([10], [10, 10], [1, 1, 1], [5, 5, 5, 5]):
        val = _longest_first_val(lens, 0.9)
        assert val, (lens, val)
        if len(lens) > 1:
            assert len(val) < len(lens), f"{lens} -> val={val} leaves train empty"


def test_deterministic_and_tie_stable():
    lens = [7] * 20 + [9] * 3
    assert _longest_first_val(lens, 0.2) == _longest_first_val(lens, 0.2)
    assert {20, 21, 22} <= _longest_first_val(lens, 0.2)   # the three 9s come first


if __name__ == "__main__":
    import traceback

    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fs:
        try:
            print(f"  {f.__name__}")
            f()
        except Exception:
            bad += 1
            print("    FAILED")
            traceback.print_exc()
    print(f"\n{len(fs) - bad}/{len(fs)} passed")
    raise SystemExit(1 if bad else 0)
