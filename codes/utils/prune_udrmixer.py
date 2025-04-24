# prune_udrmixer.py
import torch, torch_pruning as tp
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from net.model import UDRMixer                       # <-- your model
from pathlib import Path

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EXAMPLE_SHAPE = (1, 3, 512, 512)     # dummy size (crop-size); change if needed
SPARSITY      = 0.30                 # 30 % channels removed
GROUP_SIZE    = 4                    # divisible by 2×2 & 4×4 PixelShuffle
CKPT_IN       = None                 # path to pre-trained .ckpt or .pth (optional)
CKPT_OUT      = "udr_mixer_pruned.pth"

# --------------------------------------------------------------------------- #
# 0) Build / load model
model = UDRMixer(dim=64).to(DEVICE)
if CKPT_IN is not None:                       # load your Lightning ckpt / weight
    sd = torch.load(CKPT_IN, map_location=DEVICE)
    # Lightning .ckpt needs key-strip:
    if "state_dict" in sd:  sd = {k.replace("net.",""): v for k,v in sd["state_dict"].items()}
    model.load_state_dict(sd, strict=False)

dummy = torch.randn(*EXAMPLE_SHAPE, device=DEVICE)

# --------------------------------------------------------------------------- #
# 1) Build dependency graph
DG = tp.DependencyGraph().build_dependency(model, example_inputs=dummy)

# 2) Define prunable layers (skip first & last convs)
def prunable(m):
    return isinstance(m, torch.nn.Conv2d) and m.kernel_size != (1, 1)

# 3) Global magnitude-based pruner
pruner = tp.pruner.GlobalPruner(
    DG, dummy,
    pruning_ratio = SPARSITY,
    group_size    = GROUP_SIZE,
    importance    = tp.importance.MagnitudeImportance(p=1),
)

pruner.step()        # one-shot (or loop pruner.step() for iterative)
pruner.finalize()    # physically removes filters

# --------------------------------------------------------------------------- #
# 4) Report result
orig = FlopCountAnalysis(UDRMixer(dim=64), dummy).total()
new  = FlopCountAnalysis(model,              dummy).total()
print(parameter_count_table(model))
print(f"FLOPs: {orig/1e9:.2f} G  →  {new/1e9:.2f} G")

# 5) Save slim weights
torch.save(model.state_dict(), CKPT_OUT)
print(f"💾  Saved pruned weights →  {CKPT_OUT}")
