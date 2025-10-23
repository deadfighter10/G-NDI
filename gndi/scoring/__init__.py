# gndi/scoring/__init__.py
from .gndi import compute_gndi_units
from .magnitude import compute_magnitude_units
from .snip import compute_snip_units
from .grasp import compute_grasp_units
from .synflow import compute_synflow_units
from .hrank import compute_hrank_units

REGISTRY = {
    "gndi": compute_gndi_units,
    "magnitude": compute_magnitude_units,
    "snip": compute_snip_units,
    "grasp": compute_grasp_units,
    "synflow": compute_synflow_units,
    "hrank": compute_hrank_units,
}
