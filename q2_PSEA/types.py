from q2_types.feature_data import FeatureData
from qiime2.plugin import SemanticType


PSEAPairs = SemanticType("PSEAPairs")
PSEAAECounts = SemanticType("PSEAAECounts")
Spline = SemanticType("Spline", variant_of=FeatureData.field["type"])


__all__ = [
    "PSEAAECounts",
    "PSEAPairs",
    "Spline",
]
