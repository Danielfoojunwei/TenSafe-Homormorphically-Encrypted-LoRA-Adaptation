
import cv2
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt

class EmpiricalResearchVisualizer:
    """
    Generates a professional multi-panel visualization for the README.md
    based on real FastUMI Pro empirical data.
    """
    def __init__(self, metrics_path: str):
        with open(metrics_path, "r") as f:
            self.history = json.load(f)

    def generate_readme_viz(self):
        cycles = [h['cycle'] for h in self.history]
        accs = [h['global_accuracy'] for h in self.history]
        confs = [h['confidence'] for h in self.history]
        complexity = [h['data_complexity'] for h in self.history]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Global Convergence (PEFT/LoRA)
        axes[0].plot(cycles, accs, color='#2ecc71', linewidth=3, label='Global SR')
        axes[0].fill_between(cycles, 0.65, accs, color='#2ecc71', alpha=0.1)
        axes[0].set_title('Empirical Convergence (LoRA PEFT)', fontweight='bold')
        axes[0].set_xlabel('FL Cycles')
        axes[0].set_ylabel('Success Rate (SR)')
        axes[0].grid(True, alpha=0.2)
        
        # 2. Expert Gating Specificity
        axes[1].plot(cycles, confs, color='#3498db', linewidth=3, label='Expert Spec.')
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title('Gating Policy Specificity (IOSP)', fontweight='bold')
        axes[1].set_xlabel('FL Cycles')
        axes[1].set_ylabel('Expert Confidence')
        axes[1].grid(True, alpha=0.2)

        # 3. Privacy-Utility (Empirical)
        # Relationship between scene complexity and local gain
        axes[2].scatter(complexity, [h['peft_local_gain'] * 1000 for h in self.history], 
                        c=cycles, cmap='viridis', alpha=0.6)
        axes[2].set_title('Privacy-Utility Surface (Real Data)', fontweight='bold')
        axes[2].set_xlabel('Image Edge Density (Data Complexity)')
        axes[2].set_ylabel('Local Gain (x10^-3)')
        axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        viz_path = "docs/images/empirical_research_summary.png"
        os.makedirs("docs/images", exist_ok=True)
        plt.savefig(viz_path, dpi=300)
        
        # Copy to brain dir for markdown relative links
        brain_path = "C:/Users/lover/.gemini/antigravity/brain/8e552d11-36bc-4b99-b279-7a01aeab08f4/empirical_research_summary.png"
        import shutil
        shutil.copy(viz_path, brain_path)
        
        print(f"[SUCCESS] README Visualization saved to {viz_path}")

if __name__ == "__main__":
    metrics = "artifacts/fl_empirical_results.json"
    if os.path.exists(metrics):
        viz = EmpiricalResearchVisualizer(metrics)
        viz.generate_readme_viz()
    else:
        print(f"[ERROR] Metrics not found: {metrics}")
