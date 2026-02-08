
import os

file_path = r"c:\Users\lover\.gemini\antigravity\playground\void-asteroid\TenSafe-Homormorphically-Encrypted-LoRA-Adaptation\he_lora_microkernel\backend\gpu_ckks_backend.py"

with open(file_path, "r") as f:
    content = f.read()

# Inject shutdown
target_init = """    def initialize(self) -> None:
        \"\"\"Initialize (simulated) keys.\"\"\"
        self._keys_generated = True
        self._initialized = True"""

replacement_init = """    def initialize(self) -> None:
        \"\"\"Initialize (simulated) keys.\"\"\"
        self._keys_generated = True
        self._initialized = True

    def shutdown(self) -> None:
        pass"""

if target_init in content:
    content = content.replace(target_init, replacement_init)
    print("Injected shutdown method.")
else:
    print("WARNING: target_init not found.")
    # Debug: Print nearby context?
    # No, simple exit.

with open(file_path, "w") as f:
    f.write(content)
