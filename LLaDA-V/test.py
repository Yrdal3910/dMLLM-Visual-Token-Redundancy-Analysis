from safetensors.torch import safe_open
import torch, os

ckpt = "/pfs/zuhaoyang/workspace/cache/siglip2-so400m-patch14-384/model.safetensors"
bad = []
with safe_open(ckpt, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if torch.isnan(t).any() or torch.isinf(t).any():
            bad.append(k); print("BAD:", k); break
print("has_bad_in_file:", bool(bad))
print("filesize(MB):", os.path.getsize(ckpt)/1024/1024)