from fractions import Fraction

import pytest
import torch

from sqpilot.quant import CostBook, apply_from_master, encode, fingerprint, packed_bytes


@pytest.mark.parametrize("bits,qmax", [(4, 7), (8, 127)])
def test_signed_limits_ties_zero_and_partial_group(bits, qmax):
    weights = torch.zeros((2, 65))
    weights[0, :7] = torch.tensor([-qmax, qmax, 0.5, 1.5, -0.5, -1.5, 2.5])
    weights[0, -1] = -3
    codes, scales, reconstructed = encode(weights, bits)
    assert codes[0, :7].tolist() == [-qmax, qmax, 0, 2, 0, -2, 2]
    assert codes[0, -1] == -qmax
    assert not codes[1].any() and not scales[1].any()
    assert scales.dtype == torch.float16
    assert reconstructed[0, -1].item() == pytest.approx(-3, abs=0.002)
    assert packed_bytes(2, 65, bits) == (74 if bits == 4 else 138)


def test_scale_underflow_fails_closed_and_saturation():
    weight = torch.tensor([[2.0**-24, -2.0**-24]], dtype=torch.float16)
    with pytest.raises(ValueError, match="scale cannot be represented"):
        encode(weight, 8)
    weight = torch.tensor([[1.0001, -1.0001]])
    codes, _, _ = encode(weight, 8)
    assert codes.tolist() == [[127, -127]]


def test_budget_enum_exact_and_counts_fixed_cost():
    costs = CostBook(18, ((10, 14), (10, 14), (10, 14), (10, 14)))
    assert costs.total((4,)*4) == 58
    assert costs.total((8,)*4) == 74
    assert [costs.budget(f) for f in (0, .25, .50, .75, 1)] == [58,62,66,70,74]
    assert costs.budget(Fraction(1,3)) == Fraction(190,3)
    assert len(costs.feasible(.50)) == 11
    assert all(costs.total(bits) <= 66 for bits in costs.feasible(.50))


def test_restore_on_error_and_repeatability():
    model = torch.nn.Sequential(torch.nn.Linear(65, 2), torch.nn.LayerNorm(2))
    master = {name: p.detach().clone() for name,p in model.named_parameters()}
    baseline = fingerprint(model)
    with pytest.raises(RuntimeError, match="intentional"):
        with apply_from_master(model, master, ("0.weight",), 4) as metrics:
            assert metrics["0.weight"]["changed"] > 0
            first = fingerprint(model)
            assert [name for name in first if first[name] != baseline[name]] == ["0.weight"]
            raise RuntimeError("intentional")
    assert fingerprint(model) == baseline
    with apply_from_master(model, master, ("0.weight",), 4):
        assert fingerprint(model) == first
    assert fingerprint(model) == baseline
