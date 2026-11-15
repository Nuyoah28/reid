"""Fixed OSNet baseline used by v5 training-objective comparisons."""
from occlusion_v3.model import StructuralOSNet


def build_model(num_classes):
    return StructuralOSNet(num_classes, variant='baseline')
