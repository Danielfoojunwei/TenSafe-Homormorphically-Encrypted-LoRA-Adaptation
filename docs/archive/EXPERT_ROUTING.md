# TensorGuard Expert Routing Configuration

This guide documents how to customize the expert-block routing in TensorGuard's Mixture-of-Experts (MoE) architecture.

## Overview

TensorGuard v2.0 uses **Instruction-Oriented Scene Parsing (IOSP)** to route gradients to specialized experts based on task semantics. The default routing can be customized for specific robot fleets or task domains.

## Default Expert Configuration

The built-in `MoEAdapter` defines four experts:

| Expert | Block Range | Keywords |
|:-------|:------------|:---------|
| `visual_primary` | 0-3 | geometric, shapes, objects, obstacles |
| `visual_aux` | 4-5 | color, texture, depth, lighting, shadows |
| `language_semantic` | 6-7 | verbs, instructions, goal, intent, command |
| `manipulation_grasp` | 8-9 | force, torque, contact, friction, gripper |

## Configuration File

Create a `expert_config.yaml` in your project root:

```yaml
# TensorGuard Expert Routing Configuration
version: "2.0"

experts:
  visual_primary:
    blocks: [0, 1, 2, 3]
    keywords:
      - geometric
      - shapes
      - objects
      - obstacles
      - spatial
    gate_threshold: 0.15

  visual_aux:
    blocks: [4, 5]
    keywords:
      - color
      - texture
      - depth
      - lighting
      - shadows
      - material
    gate_threshold: 0.10

  language_semantic:
    blocks: [6, 7]
    keywords:
      - verbs
      - instructions
      - goal
      - intent
      - command
      - action
    gate_threshold: 0.15

  manipulation_grasp:
    blocks: [8, 9]
    keywords:
      - force
      - torque
      - contact
      - friction
      - gripper
      - pressure
    gate_threshold: 0.20

# Global settings
gating:
  method: "iosp"  # Options: iosp, learned, uniform
  sparsity: 0.15  # Experts with weight < this are excluded
  softmax_temperature: 1.0
```

## Loading Custom Configuration

```python
from tensorguard.core.adapters import MoEAdapter
import yaml

# Load custom configuration
with open("expert_config.yaml") as f:
    config = yaml.safe_load(f)

# Create adapter with custom routing
adapter = MoEAdapter()

# Override routing
adapter.expert_prototypes = {
    exp_name: exp_config["keywords"]
    for exp_name, exp_config in config["experts"].items()
}

# Override block mapping
adapter.routing = {
    exp_name: exp_config["blocks"]
    for exp_name, exp_config in config["experts"].items()
}
```

## Adding Custom Experts

To add a domain-specific expert (e.g., for surgical robotics):

```python
class SurgicalMoEAdapter(MoEAdapter):
    def __init__(self):
        super().__init__(experts=[
            "visual_primary",
            "visual_aux", 
            "language_semantic",
            "manipulation_grasp",
            "surgical_precision"  # New expert
        ])
        
        # Add keywords for the new expert
        self.expert_prototypes["surgical_precision"] = [
            "incision", "suture", "vessel", "tissue",
            "hemostasis", "retraction", "dissection"
        ]
        
        # Map to model blocks (adjust for your VLA model)
        self.routing["surgical_precision"] = [10, 11]
```

## Learned Gating (Advanced)

For production deployments, you can replace keyword-based IOSP with learned embeddings:

```python
from transformers import AutoTokenizer, AutoModel
import torch

class LearnedGatingAdapter(MoEAdapter):
    def __init__(self, encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model)
        
        # Pre-compute expert prototype embeddings
        self._expert_embeddings = self._compute_expert_embeddings()
    
    def _compute_expert_embeddings(self):
        embeddings = {}
        for expert, keywords in self.expert_prototypes.items():
            text = " ".join(keywords)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            embeddings[expert] = outputs.last_hidden_state.mean(dim=1)
        return embeddings
    
    def get_expert_gate_weights(self, task_instruction: str):
        # Encode instruction
        inputs = self.tokenizer(task_instruction, return_tensors="pt", padding=True)
        with torch.no_grad():
            task_emb = self.encoder(**inputs).last_hidden_state.mean(dim=1)
        
        # Compute cosine similarity to each expert
        weights = {}
        for expert, proto_emb in self._expert_embeddings.items():
            sim = torch.cosine_similarity(task_emb, proto_emb)
            weights[expert] = float(sim)
        
        # Softmax normalize
        import numpy as np
        vals = np.array(list(weights.values()))
        exp_vals = np.exp(vals - np.max(vals))
        norm = exp_vals / exp_vals.sum()
        
        return dict(zip(weights.keys(), norm))
```

## Debugging Expert Selection

Enable verbose logging to see expert gate decisions:

```python
import logging
logging.getLogger("tensorguard.core.adapters").setLevel(logging.DEBUG)
```

Output will show:
```
DEBUG:tensorguard.core.adapters:Task: "pick up the blue block"
DEBUG:tensorguard.core.adapters:Expert weights: visual_primary=0.42, visual_aux=0.18, language_semantic=0.25, manipulation_grasp=0.15
DEBUG:tensorguard.core.adapters:Active experts (>0.15): visual_primary, language_semantic
```
