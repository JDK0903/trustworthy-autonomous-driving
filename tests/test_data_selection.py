import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parents[1] / "scripts/prepare_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_is_stable_and_excludes_official_test():
    rgb = [f"leftImg8bit/{split}/city/city_000000_{i:06d}_leftImg8bit.png"
           for split in ("train", "val", "test") for i in range(10)]
    masks = [n.replace("leftImg8bit/", "gtFine/").replace("_leftImg8bit.png", "_gtFine_labelIds.png") for n in rgb]
    first = module.select_pairs(rgb, masks, {"train":4, "val":2}, 42)
    assert first == module.select_pairs(list(reversed(rgb)), masks, {"train":4, "val":2}, 42)
    assert len(first) == 6
    assert all(split != "test" for split, _, _ in first)
    with pytest.raises(ValueError, match="Missing annotation"):
        module.select_pairs(rgb, [], {"train":1}, 42)


def test_archive_path_cannot_escape_destination(tmp_path):
    with pytest.raises(ValueError, match="Unsafe archive path"):
        module.safe_target(tmp_path, "../outside.png")
