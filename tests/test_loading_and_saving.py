import random
from pathlib import Path
import pytest
import torch

from diff_curves import load_bpt, save_bpt

def test_bpt(tmp_path: Path):
    with pytest.raises(RuntimeError):
        save_bpt(tmp_path / "empty.bpt", torch.zeros((0, 4, 4, 3)))

    for it in range(100):
        num_patches = random.randint(1, 100)
        n = random.randint(1, 11)
        m = random.randint(1, 11)

        filepath = tmp_path / f"{it}_{n}_{m}.bpt"

        patches_expected = torch.rand((num_patches, n+1, m+1, 3))
        save_bpt(filepath, patches_expected)

        patches_actual = load_bpt(filepath)

        assert torch.allclose(patches_expected, patches_actual)