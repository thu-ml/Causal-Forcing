import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]

lmdb = pytest.importorskip("lmdb")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_long_video_dataset():
    # long_video/utils/dataset.py imports `utils.lmdb`; that name resolves to
    # long_video/utils/lmdb.py only when the process runs inside long_video/.
    # Register that exact helper module for the duration of the load.
    helper = _load(_ROOT / "long_video" / "utils" / "lmdb.py", "utils.lmdb")
    saved = sys.modules.get("utils.lmdb")
    sys.modules["utils.lmdb"] = helper
    try:
        return _load(
            _ROOT / "long_video" / "utils" / "dataset.py", "long_video_dataset"
        )
    finally:
        if saved is None:
            sys.modules.pop("utils.lmdb", None)
        else:
            sys.modules["utils.lmdb"] = saved


# Both the root project and long_video/ ship their own copy of the dataset
# helpers, and each resolves its own copy depending on the working directory.
DATASETS = [
    pytest.param(_load(_ROOT / "utils" / "dataset.py", "root_dataset"), id="root"),
    pytest.param(_load_long_video_dataset(), id="long_video"),
]


def _write_shard(path, n):
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=1 << 20)
    with env.begin(write=True) as txn:
        for i in range(n):
            txn.put(f"latents_{i}_data".encode(), np.zeros((1, 1, 1, 1, 1), dtype=np.float16).tobytes())
            txn.put(f"prompts_{i}_data".encode(), f"prompt-{i}".encode())
        txn.put(b"latents_shape", f"{n} 1 1 1 1 1".encode())
    env.close()


@pytest.mark.parametrize("dataset_module", DATASETS)
def test_sharding_dataset_honors_max_pair(tmp_path, dataset_module):
    # lmdb refuses to open the same environment twice in one process, so the
    # full and limited datasets need their own copies of the shards.
    _write_shard(tmp_path / "full" / "shard_a", 3)
    _write_shard(tmp_path / "full" / "shard_b", 2)
    full = dataset_module.ShardingLMDBDataset(str(tmp_path / "full"))
    assert len(full) == 5

    _write_shard(tmp_path / "limited" / "shard_a", 3)
    _write_shard(tmp_path / "limited" / "shard_b", 2)
    limited = dataset_module.ShardingLMDBDataset(str(tmp_path / "limited"), max_pair=3)
    assert len(limited) == 3
    assert limited[2]["prompts"] == "prompt-2"


@pytest.mark.parametrize("dataset_module", DATASETS)
def test_sharding_dataset_keeps_full_length_by_default(tmp_path, dataset_module):
    _write_shard(tmp_path / "shard_a", 4)
    dataset = dataset_module.ShardingLMDBDataset(str(tmp_path))
    assert len(dataset) == 4
