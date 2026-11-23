import pytest

from tools.p0_model import POSITION_GATE, PROJECTIONS, cost, inventory


@pytest.fixture
def wavlm_shapes():
    return {
        f"encoder.layers.{index}.{name}": shape
        for index in range(12)
        for name, shape in (PROJECTIONS | {POSITION_GATE: (8, 64)}).items()
    }


def test_inventory_preserves_both_eligibility_readings(wavlm_shapes):
    result = inventory(wavlm_shapes)
    assert result["status"] == "BLOCKED"
    readings = result["interpretations"]
    assert readings["six_named"]["eligible_matrices"] == 72
    assert readings["seven_linear_including_gate"]["eligible_matrices"] == 84
    assert readings["six_named"]["excluded_elements"] == 6_144


@pytest.mark.parametrize("name", ["attention.q_proj.weight", POSITION_GATE])
def test_inventory_fails_when_a_block_matrix_is_missing(wavlm_shapes, name):
    del wavlm_shapes[f"encoder.layers.7.{name}"]
    with pytest.raises(ValueError, match="Missing or unexpected shape"):
        inventory(wavlm_shapes)


@pytest.mark.parametrize("bits, expected", [(4, 74), (8, 138)])
def test_packed_cost_includes_scale_and_rounds_partial_group(bits, expected):
    assert cost((2, 65), bits) == expected
