"""
Benchmark Reporting Module
Generates HTML dashboard from JSONL artifacts.
"""

import os
import json
import glob
import logging
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.output_file = os.path.join(artifacts_dir, "report.html")

    def load_metrics(self):
        data = []
        # Load Microbench
        for f in glob.glob(os.path.join(self.artifacts_dir, "metrics/micro_bench_*.jsonl")):
            with open(f, 'r') as fh:
                for line in fh:
                    data.append(json.loads(line))
                    
        # Load Privacy
        privacy_data = []
        p_file = os.path.join(self.artifacts_dir, "privacy/inversion_results.json")
        if os.path.exists(p_file):
            with open(p_file, 'r') as fh:
                privacy_data = json.load(fh)

        # Load Robustness
        robust_data = {}
        r_file = os.path.join(self.artifacts_dir, "robustness/byzantine_results.json")
        if os.path.exists(r_file):
             with open(r_file, 'r') as fh:
                robust_data = json.load(fh)

        return data, privacy_data, robust_data

    def generate_json(self, micro, privacy, robust):
        import platform
        import subprocess
        import hashlib
        import uuid
        
        # Gather Environment Info
        env_info = {
            "os": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        }
        
        # Gather Git Info
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Could not get git commit: {e}")
            commit = "unknown"

        # Calculate Latency Stats from Microbench
        latencies = []
        for m in micro:
            mets = m.get('metrics', {})
            # Try various common keys
            l = mets.get('latency_ms') or mets.get('p50_latency_sec', 0) * 1000.0 or mets.get('latency_sec', 0) * 1000.0
            if l:
                latencies.append(l)

        # Use statistics module for efficient percentile calculation
        metrics_summary = {
            "privacy_mse": privacy[0]['metrics']['mse'] if privacy else 0.0,
            "robustness_success": robust.get('success', False) if robust else None,
            "latency_p50": statistics.median(latencies) if latencies else None,
            "latency_avg": statistics.mean(latencies) if latencies else None
        }

        # Construct Report Object
        report_id = str(uuid.uuid4())
        report_data = {
            "schema": "tensorguard.artifact.run.v1",
            "run_id": report_id,
            "timestamp": datetime.now().isoformat(),
            "sdk_version": "2.1.0",
            "git_commit": commit,
            "environment": env_info,
            "metrics": metrics_summary,
            "details": {
                "micro": micro,
                "privacy": privacy,
                "robustness": robust
            }
        }
        
        # Save JSON
        json_path = os.path.join(self.artifacts_dir, "report.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Hash HTML if exists
        artifacts = {}
        if os.path.exists(self.output_file):
            with open(self.output_file, 'rb') as f:
                artifacts["report.html"] = hashlib.sha256(f.read()).hexdigest()
                
        # Hash JSON itself (meta-circular but useful if signed separately)
        # We can't hash the file we just wrote perfectly if we want to include the hash inside it.
        # So we store the artifacts dictionary separate or strictly as "other artifacts".
        
        # Update JSON with available artifact hashes
        report_data["artifacts_hashes"] = artifacts
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"JSON Report generated: {json_path}")
        return json_path

    def generate_signed_evidence(self, report_path: str):
        """Generate a Signed Evidence Event for the report."""
        import hashlib
        from tensorguard.evidence.schema import EvidenceEvent, EventType
        from tensorguard.evidence.signing import sign_event, generate_keypair
        from tensorguard.evidence.store import get_store
        
        # 1. Calc Hash
        with open(report_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            
        # 2. Create Event
        evt = EvidenceEvent(
            event_type=EventType.BENCH_REPORT_GENERATED,
            subject={"report_ref": report_path},
            artifacts=[{"name": "report.json", "sha256": h}],
            result={"status": "success"}
        ).model_dump()
        
        # 3. Sign (Use Ephemeral Key for MVP if no CI key)
        # In real CI, load_private_key from env.
        priv, pub = generate_keypair()
        evt = sign_event(evt, priv, "ci_ephemeral_key")
        
        # 4. Save
        store = get_store()
        out_path = store.save_event(evt)
        print(f"Signed Evidence Generated: {out_path}")
        return out_path

    def generate(self):
        micro, privacy, robust = self.load_metrics()
        
        # ... (Existing HTML Generation Logic - kept intact) ...
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorGuard Benchmark Report</title>
            <style>
                body {{ font-family: sans-serif; margin: 2em; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>TensorGuard Benchmark Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>

            <h2>1. Privacy Evaluation (Gradient Inversion)</h2>
            <table>
                <tr>
                    <th>Scenario</th>
                    <th>MSE (Lower is Better)</th>
                    <th>RRE (Higher is Better Privacy)</th>
                    <th>Simulated Attack PSNR</th>
                </tr>
        """
        
        for item in privacy:
            m = item['metrics']
            html += f"""
                <tr>
                    <td>{item['scenario']}</td>
                    <td>{m['mse']:.6f}</td>
                    <td>{m['rre']:.6f}</td>
                    <td>{m['simulated_attack_psnr']:.2f} dB</td>
                </tr>
            """
            
        html += """
            </table>

            <h2>2. Robustness (Byzantine Resilience)</h2>
        """
        
        if robust:
            status = "<span class='pass'>PASSED</span>" if robust.get("success") else "<span class='fail'>FAILED</span>"
            html += f"""
            <p>Status: {status}</p>
            <p>Expected Outliers: {robust.get('expected_outliers')}</p>
            <p>Detected Outliers: {robust.get('detected_outliers')}</p>
            <p>False Positives: {robust.get('false_positives')}</p>
            <p>Detection Time: {robust.get('detection_time_sec'):.4f}s</p>
            """
        else:
            html += "<p>No robustness results found.</p>"
            
        html += """
            <h2>3. Microbenchmarks</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Timestamp</th>
                    <th>Metrics</th>
                </tr>
        """
        
        for entry in micro:
            metrics_str = ", ".join([f"{k}={v}" for k,v in entry['metrics'].items()])
            html += f"""
                <tr>
                    <td>{entry['test_id']}</td>
                    <td>{datetime.fromtimestamp(entry['timestamp']).isoformat()}</td>
                    <td>{metrics_str}</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>4. Compliance Evidence</h2>
            <p>See <code>evidence_pack/</code> directory for SOC2/GDPR/HIPAA artifacts.</p>
        </body>
        </html>
        """
        
        with open(self.output_file, 'w') as f:
            f.write(html)
        print(f"Report generated: {self.output_file}")
        
        # NEW: Generate JSON as well
        json_path = self.generate_json(micro, privacy, robust)
        
        # NEW: Generate Signed Evidence
        self.generate_signed_evidence(json_path)


def run_report(args):
    gen = ReportGenerator()
    gen.generate()
