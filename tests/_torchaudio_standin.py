from __future__ import annotations

import numpy as np
import soundfile as sf


def load(uri, *args, **kwargs):
    import torch

    data, sample_rate = sf.read(str(uri), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate


def save(uri, src, sample_rate, *args, **kwargs):
    path = str(uri)
    is_flac = path.lower().endswith(".flac")
    subtype = "PCM_16" if is_flac else "FLOAT"
    sf.write(path, src.detach().cpu().numpy().T, int(sample_rate), subtype=subtype)
