from types import SimpleNamespace

import numpy as np
import pytest

from environment.benchmarks.robodojo.compat.isaac601 import rgba_batch, update_without_physics


def test_capture_keeps_channel_order_dimensions_and_owns_frame():
    source = np.array([[[255, 1, 2, 9], [3, 254, 4, 8]],
                       [[5, 6, 253, 7], [10, 11, 12, 6]]], dtype=np.uint8)
    result = rgba_batch(source.reshape(-1), 2, 2)
    assert result.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(result[0], source)
    source.fill(0)
    assert result[0, 0, 0].tolist() == [255, 1, 2, 9]
    with pytest.raises(RuntimeError):
        rgba_batch(np.array([], dtype=np.uint8), 2, 2)


@pytest.mark.parametrize("fails", [False, True])
def test_render_does_not_step_physics_and_restores_setting_on_error(fails):
    key = "/app/player/playSimulations"
    values = {key: True}
    settings = SimpleNamespace(get=values.get, set=values.__setitem__)
    def update():
        assert values[key] is False
        if fails:
            raise ValueError("render failed")
    if fails:
        with pytest.raises(ValueError):
            update_without_physics(SimpleNamespace(update=update), settings)
    else:
        update_without_physics(SimpleNamespace(update=update), settings)
    assert values[key] is True


def test_conveyor_scalar_preserves_authored_physical_velocity():
    from environment.benchmarks.robodojo.compat.conveyor import scalar_for_surface
    import pytest
    assert scalar_for_surface([-1, 0, 0], [-.1, 0, 0]) == pytest.approx(.1)
    with pytest.raises(ValueError):
        scalar_for_surface([-1, 0, 0], [0, .1, 0])
