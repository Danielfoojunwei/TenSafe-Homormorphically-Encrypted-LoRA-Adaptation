"""
Robot Action Classifier - XGBoost-based Action Identification

Implements the classification pipeline from arXiv:2312.06802 Section 6.
Achieves ~97% accuracy on robot action identification using signal-processing
derived features.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Optional XGBoost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

# Optional sklearn imports for metrics
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, 
        classification_report, roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class EvaluationReport:
    """Comprehensive evaluation metrics."""
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    class_names: List[str]
    feature_importance: Dict[str, float]
    roc_auc: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_per_class": self.f1_per_class,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_names": self.class_names,
            "feature_importance": self.feature_importance,
            "roc_auc": self.roc_auc,
        }
    
    def save_json(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_html(self, path: Path) -> None:
        """Save report as HTML."""
        html = self._generate_html()
        with open(path, 'w') as f:
            f.write(html)
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        # Confusion matrix as HTML table
        cm_html = "<table class='confusion-matrix'><tr><th></th>"
        for name in self.class_names:
            cm_html += f"<th>{name}</th>"
        cm_html += "</tr>"
        
        for i, name in enumerate(self.class_names):
            cm_html += f"<tr><th>{name}</th>"
            for j in range(len(self.class_names)):
                val = self.confusion_matrix[i, j]
                cm_html += f"<td>{val}</td>"
            cm_html += "</tr>"
        cm_html += "</table>"
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>RTPL Attack Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
        .confusion-matrix {{ border-collapse: collapse; }}
        .confusion-matrix th, .confusion-matrix td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .confusion-matrix th {{ background: #4CAF50; color: white; }}
        .feature-bar {{ background: #2196F3; height: 20px; margin: 2px 0; }}
    </style>
</head>
<body>
    <h1>🤖 RTPL Attack Evaluation Report</h1>
    <p>Traffic Analysis Attack Reproduction (arXiv:2312.06802)</p>
    
    <div class="metric">
        <h3>Overall Accuracy: {self.accuracy:.1%}</h3>
        <p>F1 Macro: {self.f1_macro:.3f}</p>
    </div>
    
    <div class="metric">
        <h3>Confusion Matrix</h3>
        {cm_html}
    </div>
    
    <div class="metric">
        <h3>Per-Class F1 Scores</h3>
        <ul>
        {"".join(f"<li>{k}: {v:.3f}</li>" for k, v in self.f1_per_class.items())}
        </ul>
    </div>
    
    <div class="metric">
        <h3>Top 10 Feature Importance</h3>
        <ul>
        {"".join(f"<li>{k}: {v:.4f}</li>" for k, v in sorted(self.feature_importance.items(), key=lambda x: -x[1])[:10])}
        </ul>
    </div>
</body>
</html>
"""


class RobotActionClassifier:
    """
    XGBoost classifier for robot action identification.
    
    Paper: "We employ XGBoost given that it is known to outperform other
    classical machine learning models in both computational speed and
    model performance, and has been successfully used for traffic analysis."
    """
    
    # Default robot actions (paper's 4-action closed-world)
    DEFAULT_ACTIONS = [
        "pick_and_place",
        "pour_water", 
        "press_key",
        "toggle_switch"
    ]
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            class_names: Names of action classes
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            random_state: Random seed for reproducibility
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.class_names = class_names or self.DEFAULT_ACTIONS
        self.n_classes = len(self.class_names)
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='multi:softmax',
            num_class=self.n_classes,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.feature_names: Optional[List[str]] = None
        self._is_trained = False
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Names of features
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        self.feature_names = feature_names
        
        if validation_split > 0 and HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
        else:
            self.model.fit(X, y, verbose=False)
            val_acc = 1.0  # No validation
        
        self._is_trained = True
        
        return {
            "validation_accuracy": val_acc,
            "n_samples": len(X),
            "n_features": X.shape[1],
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict action classes."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> EvaluationReport:
        """
        Comprehensive evaluation on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            EvaluationReport with all metrics
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for evaluation")
        
        y_pred = self.predict(X)
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        
        # Per-class F1
        f1_per = f1_score(y, y_pred, average=None)
        f1_per_class = {
            self.class_names[i]: float(f1_per[i]) 
            for i in range(len(self.class_names))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance
        importance = self.feature_importance()
        
        # ROC AUC (if binary or multi-class with probabilities)
        roc_auc = None
        try:
            y_proba = self.predict_proba(X)
            if self.n_classes == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
        except Exception:
            pass
        
        return EvaluationReport(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_per_class=f1_per_class,
            confusion_matrix=cm,
            class_names=self.class_names,
            feature_importance=importance,
            roc_auc=roc_auc
        )
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Paper Section 6.4: "Feature importance is determined based on the
        average gain, which measures the improvement in loss function."
        """
        if not self._is_trained:
            return {}
        
        importance = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importance):
            return {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            return {
                f"feature_{i}": float(imp) 
                for i, imp in enumerate(importance)
            }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Returns mean and std of accuracy across folds.
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for cross-validation")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }
    
    def save(self, path: Path) -> None:
        """Save trained model."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        self.model.save_model(str(path))
    
    def load(self, path: Path) -> None:
        """Load trained model."""
        self.model.load_model(str(path))
        self._is_trained = True
