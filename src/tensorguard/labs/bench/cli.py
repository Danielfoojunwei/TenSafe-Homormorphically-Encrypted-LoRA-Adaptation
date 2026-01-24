"""
tg-bench: TensorGuard Benchmark CLI
"""

import argparse
import sys
from .micro import run_micro
from .privacy.inversion import run_privacy
from .robustness.byzantine import run_robustness
from .compliance.evidence import run_evidence
from .reporting import run_report
import requests
import json
import os

def run_upload(args):
    """Upload benchmark run to Platform."""
    server = args.server.rstrip('/')
    
    # 1. Load artifact
    json_path = "artifacts/report.json"
    if not os.path.exists(json_path):
        print("Error: artifacts/report.json not found. Run 'report' command first.")
        return

    print(f"Uploading run to {server}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 2. Register Run
    try:
        res = requests.post(f"{server}/api/v1/runs", json=data)
        res.raise_for_status()
        run = res.json()
        run_id = run['run_id']
        print(f"Run Registered: {run_id}")
        
        # 3. Upload Artifact
        with open(json_path, 'rb') as f:
            files = {'file': ('report.json', f, 'application/json')}
            res2 = requests.post(f"{server}/api/v1/runs/{run_id}/artifacts", files=files, params={'artifact_type': 'report.json'})
            res2.raise_for_status()
        print("Artifact Uploaded.")
        
        # 4. Evaluate
        print(f"Evaluating against policy: {args.policy}")
        res3 = requests.post(f"{server}/api/v1/runs/{run_id}/evaluate", params={'pack_id': args.policy})
        res3.raise_for_status()
        result = res3.json()
        
        color = "\033[92m" if result['passed'] else "\033[91m"
        reset = "\033[0m"
        print(f"\n{color}Trust Score: {result['score']:.1f}/100{reset}")
        print(f"Status: {'PASSED' if result['passed'] else 'FAILED'}")
        
    except Exception as e:
        print(f"Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="TensorGuard Benchmark Suite")
    subparsers = parser.add_subparsers(dest="command", help="Benchmark Command")
    
    # ... existing commands ...
    
    # 7. Upload
    upload_parser = subparsers.add_parser("upload", help="Upload run to Platform")
    upload_parser.add_argument("--server", default="http://localhost:8000", help="Platform URL")
    upload_parser.add_argument("--policy", default="soc2-evidence-pack", help="Policy Pack ID")

    # 1. Micro
    micro_parser = subparsers.add_parser("micro", help="Run microbenchmarks")

    
    # 2. UpdatePkg (Pipeline) - MVP stub
    update_parser = subparsers.add_parser("updatepkg", help="Run client pipeline bench (not yet implemented)")
    
    # 3. Privacy
    privacy_parser = subparsers.add_parser("privacy", help="Run privacy evaluation")

    # 4. Robustness
    robustness_parser = subparsers.add_parser("robustness", help="Run robustness/byzantine tests")
    
    # 5. Evidence
    evidence_parser = subparsers.add_parser("evidence", help="Generate compliance evidence")
    
    # 6. Report
    report_parser = subparsers.add_parser("report", help="Generate HTML report")

    args = parser.parse_args()
    
    if args.command == "micro":
        run_micro(args)
    elif args.command == "updatepkg":
        print("Not implemented yet")
    elif args.command == "privacy":
        run_privacy(args)
    elif args.command == "robustness":
        run_robustness(args)
    elif args.command == "evidence":
        run_evidence(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "upload":
        run_upload(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
