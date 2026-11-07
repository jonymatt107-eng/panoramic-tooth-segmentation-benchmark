"""Protocol invariants: leakage, metric edge cases, paired noise and aggregation."""
import numpy as np
import pandas as pd
import pytest
import torch
from config import Config
from dataset import make_splits
from losses_and_metrics import binary_metrics, CombinedLoss
from robustness_eval import conditions, corrupt, corruption_rng
from generate_paper_artifacts import patient_scores, estimate


def test_patient_partition_and_fold_separation() -> None:
    frame = pd.DataFrame({"patient_id": [f"p{i//2}" for i in range(40)]})
    result = make_splits(frame, Config())
    assert result.groupby("patient_id").split.nunique().max() == 1
    dev = result[result.split == "dev"]
    assert dev.groupby("patient_id").fold.nunique().max() == 1
    for fold in range(5):
        assert not set(dev[dev.fold == fold].patient_id) & set(dev[dev.fold != fold].patient_id)
    assert make_splits(frame, Config()).equals(result)


def test_explicit_overlap_rejected() -> None:
    frame = pd.DataFrame({"patient_id": ["same", "same"], "split": ["dev", "test"]})
    with pytest.raises(ValueError, match="overlaps"):
        make_splits(frame, Config())


def test_metric_edge_cases_and_hand_computation() -> None:
    logits = torch.tensor([[[[10., 10., -10., -10.]]]])
    target = torch.tensor([[[[1., 0., 1., 0.]]]])
    row = binary_metrics(logits, target)[0]
    assert row["dice"] == .5 and row["iou"] == pytest.approx(1/3)
    assert row["sensitivity"] == row["specificity"] == row["precision"] == .5
    row = binary_metrics(torch.full((1, 1, 2, 2), -10.), torch.zeros(1, 1, 2, 2))[0]
    assert row["dice"] == 1 and np.isnan(row["sensitivity"]) and np.isnan(row["precision"])
    assert row["specificity"] == 1


def test_loss_stable_with_extreme_logits() -> None:
    x = torch.tensor([[[[1000., -1000.]]]], requires_grad=True)
    loss = CombinedLoss()(x, torch.tensor([[[[0., 1.]]]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_corruption_range_shape_and_pairing() -> None:
    image = np.random.default_rng(2).uniform(.2, .8, (64, 64, 3)).astype(np.float32)
    for kind, severity, value in conditions():
        a = corrupt(image, kind, value, corruption_rng(42, "id", kind, 0), severity)
        b = corrupt(image, kind, value, corruption_rng(42, "id", kind, 0), severity)
        assert np.array_equal(a, b) and a.shape == image.shape
        assert 0 <= a.min() <= a.max() <= 1
    assert np.array_equal(image, corrupt(image, "clean", 0, np.random.default_rng(1)))


def test_patient_aggregation_does_not_overweight_images_or_folds() -> None:
    rows = []
    for fold in range(2):
        for patient, images, dice in [("a", 3, 1.), ("b", 1, 0.)]:
            for image in range(images):
                for repeat in range(2):
                    rows.append(dict(model="m", fold=fold, patient_id=patient, image_id=f"{patient}{image}",
                        corruption="clean", severity=0, repeat=repeat, dice=dice, iou=dice,
                        sensitivity=dice, specificity=dice, precision=dice, foreground=1, fp=1, tn=1))
    scores = patient_scores(pd.DataFrame(rows))
    assert len(scores) == 2
    assert estimate(scores.dice.to_numpy(), 100, 42)["mean"] == .5


def test_mit_compatibility_output_and_gradient_equivalence() -> None:
    from copy import deepcopy
    from segmentation_models_pytorch.encoders.mix_transformer import Attention
    from mps_compat import enable_deterministic_mit
    for ratio in (1, 2):
        torch.manual_seed(42)
        original = Attention(dim=32, num_heads=1, sr_ratio=ratio).eval()
        patched = enable_deterministic_mit(deepcopy(original))
        x = torch.randn(2, 64, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a, b = original(x, 8, 8), patched(y, 8, 8)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(original.parameters(), patched.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        assert original.state_dict().keys() == patched.state_dict().keys()


def test_bilinear_adjoint_matches_native_gradient() -> None:
    from mps_compat import DeterministicBilinear
    for align in (True, False):
        for shape, size in [((2, 3), (16, 12)), ((8, 8), (16, 16)), ((16, 16), (64, 64)), ((16, 16), (8, 8))]:
            x = torch.randn(2, 3, *shape, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            a = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=align)
            b = DeterministicBilinear.apply(y, *size, align)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            gradient = torch.randn_like(a)
            a.backward(gradient); b.backward(gradient)
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=3e-5)


def test_image_and_mask_validation(tmp_path) -> None:
    import cv2
    from dataset import read_pair
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    cv2.imwrite(str(image_path), np.arange(64, dtype=np.uint8).reshape(8, 8))
    cv2.imwrite(str(mask_path), np.eye(8, dtype=np.uint8)*255)
    image, mask = read_pair(str(image_path), str(mask_path))
    assert image.shape == (8, 8, 3) and set(np.unique(mask)) == {0., 1.}
    cv2.imwrite(str(mask_path), np.full((8, 8), 127, np.uint8))
    with pytest.raises(ValueError, match="0/1 or 0/255"):
        read_pair(str(image_path), str(mask_path))
    cv2.imwrite(str(image_path), np.full((8, 8), 32768, np.uint16))
    with pytest.raises(ValueError, match="8-bit"):
        read_pair(str(image_path), str(mask_path))


def test_letterbox_preserves_shape_and_exact_mask_padding() -> None:
    from dataset import letterbox_pair, transforms
    cfg = Config(image_size=512)
    image = np.full((120, 300, 3), 180, np.uint8)
    mask = np.ones((120, 300), np.float32)
    x, y, valid = letterbox_pair(image, mask, cfg)
    assert x.shape == (512, 512, 3) and y.shape == valid.shape == (512, 512)
    assert np.array_equal(y, valid)
    content = np.argwhere(valid)
    height, width = content.max(0)-content.min(0)+1
    assert width == 512 and abs(height/width - 120/300) < 1/512
    assert np.all(x[valid == 0] == 0) and np.all(y[valid == 0] == 0)
    transformed = transforms(cfg)(image=image, mask=mask, valid=mask)
    assert np.array_equal(transformed['mask'].numpy(), transformed['valid'].numpy())
    assert np.array_equal(transformed['mask'].numpy(), y)


def test_padding_neither_changes_loss_nor_metrics() -> None:
    logits = torch.tensor([[[[8., -8.], [-8., 8.]]]], requires_grad=True)
    mask = torch.tensor([[[[1., 0.], [0., 1.]]]])
    native = binary_metrics(logits, mask)[0]
    padded = torch.nn.functional.pad(logits, (2, 2, 2, 2), value=100)
    target = torch.nn.functional.pad(mask, (2, 2, 2, 2))
    valid = torch.nn.functional.pad(torch.ones_like(mask), (2, 2, 2, 2))
    assert binary_metrics(padded, target, valid=valid)[0] == native
    torch.testing.assert_close(CombinedLoss()(logits, mask), CombinedLoss()(padded, target, valid))
    CombinedLoss()(padded, target, valid).backward()
    assert torch.isfinite(logits.grad).all()


def test_frozen_fold_assignments_preserved() -> None:
    frame = pd.DataFrame({'patient_id': [f'p{i}' for i in range(12)],
                          'split': ['dev']*10+['test']*2, 'fold': [i%5 for i in range(10)]+[-1,-1]})
    result = make_splits(frame, Config())
    assert result.fold.tolist() == frame.fold.tolist()
