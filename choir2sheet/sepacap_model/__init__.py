"""SepACap network (SepReformer adapted for a cappella separation).

Vendored from https://github.com/ETH-DISCO/SepACap (Lanzendörfer, Pinkl,
Grötschla, "Source Separation for A Cappella Music", ICASSP 2026), which
derives from SepReformer (https://github.com/dmlguq456/SepReformer).
Only the inference model is kept; training/logging code is dropped.
Weights: https://huggingface.co/Tino3141/sepacap (MIT).
"""

from .model import Model

__all__ = ["Model"]
