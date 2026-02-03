"""
Pruning parameter dataclass for AdaPrune.

Defines common parameters used in tree pruning for both scikit-learn
and XGBoost-style decision tree ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class PruningParams:
    """
    Container for pruning-related hyperparameters.

    Attributes:
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        min_samples_split: Minimum samples required to split.
        max_features: Fraction of features to consider at split (None for all).
        min_impurity_decrease: Minimum impurity decrease required to split.
        min_child_weight: XGBoost min child weight.
        gamma: XGBoost gamma (min split loss).
        reg_alpha: XGBoost L1 regularization.
        reg_lambda: XGBoost L2 regularization.
    """

    max_depth: int = 10
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    max_features: Optional[float] = None
    min_impurity_decrease: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pruning parameters to a plain dictionary.

        Returns:
            A dictionary representation of the parameters.
        """
        return asdict(self)

    def copy(self) -> "PruningParams":
        """
        Create a shallow copy of the parameter object.

        Returns:
            A new PruningParams instance with identical values.
        """
        return PruningParams(**self.to_dict())

    def to_sklearn_params(self) -> Dict[str, Any]:
        """
        Convert to scikit-learn compatible parameter dictionary.

        Returns:
            Dictionary with scikit-learn style parameter names.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "min_impurity_decrease": self.min_impurity_decrease,
        }

    def to_xgboost_params(self) -> Dict[str, Any]:
        """
        Convert to XGBoost compatible parameter dictionary.

        Returns:
            Dictionary with XGBoost style parameter names.
        """
        return {
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PruningParams":
        """
        Build a PruningParams instance from a dictionary.

        Args:
            d: Dictionary of pruning parameters.

        Returns:
            A PruningParams instance.
        """
        return cls(**d)