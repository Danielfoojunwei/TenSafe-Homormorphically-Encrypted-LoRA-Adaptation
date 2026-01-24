
import json
import hashlib
from pathlib import Path

def generate_fixtures():
    fixtures_dir = Path("tests/qa/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Low Novelty Fixture
    low_novelty = {
        "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "novelty_score": 0.05,
        "should_train": False
    }
    with open(fixtures_dir / "low_novelty.json", "w") as f:
        json.dump(low_novelty, f, indent=2)

    # 2. High Novelty Fixture
    high_novelty = {
        "content_hash": hashlib.sha256(b"new-content").hexdigest(),
        "novelty_score": 0.85,
        "should_train": True
    }
    with open(fixtures_dir / "high_novelty.json", "w") as f:
        json.dump(high_novelty, f, indent=2)
        
    print("Fixtures generated in tests/qa/fixtures/")

if __name__ == "__main__":
    generate_fixtures()
