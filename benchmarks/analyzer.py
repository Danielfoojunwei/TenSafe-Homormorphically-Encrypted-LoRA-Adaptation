"""
Benchmark Results Analyzer

Processes benchmark results, generates plots, and creates reports.
Compares results against academic benchmarks and identifies bottlenecks.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plot generation disabled")


# Reference benchmarks from literature (see docs/research_benchmark_summary.md)
REFERENCE_BENCHMARKS = {
    "opentelemetry_collector": {
        "throughput_rps": 5000,  # Single instance
        "latency_p50_ms": 5,
        "latency_p99_ms": 50,
        "source": "Becker (2023)"
    },
    "fastapi_with_db": {
        "throughput_rps": 3000,
        "latency_p50_ms": 10,
        "latency_p99_ms": 100,
        "source": "TechEmpower Benchmarks"
    },
    "flink_streaming": {
        "events_per_second": 500000,
        "latency_p99_ms": 100,
        "source": "Yahoo Streaming Benchmark"
    },
}


class BenchmarkAnalyzer:
    """
    Analyzes benchmark results and generates reports.
    """

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results: Dict[str, Any] = {}
        self.analysis: Dict[str, Any] = {}

        self._load_results()

    def _load_results(self):
        """Load benchmark results from JSON file."""
        if self.results_path.exists():
            with open(self.results_path) as f:
                self.results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of benchmark results."""
        self.analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "comparison": self._compare_to_references(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations(),
        }
        return self.analysis

    def _compare_to_references(self) -> Dict[str, Any]:
        """Compare results against academic/industry benchmarks."""
        comparison = {}

        # Compare HTTP throughput
        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if http_results and "summary" in http_results:
            avg_rps = http_results["summary"].get("avg_rps", 0)
            avg_p95 = http_results["summary"].get("avg_p95_latency_ms", 0)

            fastapi_ref = REFERENCE_BENCHMARKS["fastapi_with_db"]

            comparison["http_api"] = {
                "measured": {
                    "throughput_rps": avg_rps,
                    "latency_p95_ms": avg_p95,
                },
                "reference": fastapi_ref,
                "throughput_ratio": avg_rps / fastapi_ref["throughput_rps"] if fastapi_ref["throughput_rps"] > 0 else 0,
                "within_order_of_magnitude": 0.1 <= avg_rps / fastapi_ref["throughput_rps"] <= 10 if fastapi_ref["throughput_rps"] > 0 else False,
            }

        # Compare telemetry throughput
        telemetry_results = self.results.get("benchmarks", {}).get("telemetry_ingest", {})
        if telemetry_results and "metrics" in telemetry_results:
            metrics = telemetry_results["metrics"]
            events_per_sec = metrics.get("throughput", {}).get("events_per_second", 0)
            p95_latency = metrics.get("latency", {}).get("p95_ms", 0)

            otel_ref = REFERENCE_BENCHMARKS["opentelemetry_collector"]

            comparison["telemetry"] = {
                "measured": {
                    "events_per_second": events_per_sec,
                    "latency_p95_ms": p95_latency,
                },
                "reference": otel_ref,
                "throughput_ratio": events_per_sec / otel_ref["throughput_rps"] if otel_ref["throughput_rps"] > 0 else 0,
                "within_order_of_magnitude": 0.1 <= events_per_sec / otel_ref["throughput_rps"] <= 10 if otel_ref["throughput_rps"] > 0 else False,
            }

        return comparison

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from results."""
        bottlenecks = []

        # Check HTTP endpoints for slow ones
        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if http_results and "endpoints" in http_results:
            for path, metrics in http_results["endpoints"].items():
                latency = metrics.get("latency", {})
                p95 = latency.get("p95_ms", 0)
                p99 = latency.get("p99_ms", 0)

                # High p99/p95 ratio indicates tail latency issues
                if p95 > 0 and p99 / p95 > 3:
                    bottlenecks.append({
                        "type": "tail_latency",
                        "component": "http_api",
                        "endpoint": path,
                        "severity": "medium",
                        "detail": f"p99 ({p99:.0f}ms) is {p99/p95:.1f}x p95 ({p95:.0f}ms)",
                    })

                # High absolute latency
                if p95 > 500:
                    bottlenecks.append({
                        "type": "high_latency",
                        "component": "http_api",
                        "endpoint": path,
                        "severity": "high",
                        "detail": f"p95 latency is {p95:.0f}ms (target: <200ms)",
                    })

        # Check resource utilization
        resource_metrics = None

        # Try to get resources from HTTP or telemetry results
        for bench_name in ["http_api", "telemetry_ingest"]:
            bench = self.results.get("benchmarks", {}).get(bench_name, {})
            if "resources" in bench:
                resource_metrics = bench["resources"]
                break

        if resource_metrics:
            cpu = resource_metrics.get("cpu", {})
            memory = resource_metrics.get("memory", {})

            if cpu.get("max_percent", 0) > 90:
                bottlenecks.append({
                    "type": "cpu_saturation",
                    "component": "system",
                    "severity": "high",
                    "detail": f"CPU peaked at {cpu['max_percent']:.0f}%",
                })

            if memory.get("max_percent", 0) > 90:
                bottlenecks.append({
                    "type": "memory_pressure",
                    "component": "system",
                    "severity": "high",
                    "detail": f"Memory peaked at {memory['max_percent']:.0f}%",
                })

        # Check telemetry acceptance rate
        telemetry_results = self.results.get("benchmarks", {}).get("telemetry_ingest", {})
        if telemetry_results and "metrics" in telemetry_results:
            metrics = telemetry_results["metrics"]
            acceptance_rate = metrics.get("acceptance_rate", 1.0)

            if acceptance_rate < 0.99:
                bottlenecks.append({
                    "type": "data_loss",
                    "component": "telemetry",
                    "severity": "medium" if acceptance_rate > 0.95 else "high",
                    "detail": f"Acceptance rate is {acceptance_rate:.1%} (target: >99%)",
                })

        return bottlenecks

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        for bottleneck in self.analysis.get("bottlenecks", []):
            bn_type = bottleneck.get("type", "")

            if bn_type == "tail_latency":
                recommendations.append({
                    "priority": "medium",
                    "area": "Database",
                    "recommendation": "Investigate slow queries. Consider adding indexes for commonly filtered columns (fleet_id, timestamp). Enable query logging to identify N+1 patterns.",
                })

            if bn_type == "high_latency":
                endpoint = bottleneck.get("endpoint", "")
                recommendations.append({
                    "priority": "high",
                    "area": "API",
                    "recommendation": f"Profile {endpoint} endpoint. Consider caching frequently accessed data, implementing pagination, or using async database operations.",
                })

            if bn_type == "cpu_saturation":
                recommendations.append({
                    "priority": "high",
                    "area": "Scaling",
                    "recommendation": "CPU is saturated. Consider horizontal scaling (more workers/replicas), optimizing hot code paths, or offloading computation to background workers.",
                })

            if bn_type == "memory_pressure":
                recommendations.append({
                    "priority": "high",
                    "area": "Memory",
                    "recommendation": "High memory usage detected. Review object lifecycles, implement streaming for large payloads, consider memory profiling to find leaks.",
                })

            if bn_type == "data_loss":
                recommendations.append({
                    "priority": "high",
                    "area": "Reliability",
                    "recommendation": "Event rejection detected. Review batch size limits, implement client-side retry with backoff, consider async ingestion with acknowledgment.",
                })

        # Add general recommendations based on comparison
        comparison = self.analysis.get("comparison", {})

        if comparison.get("http_api", {}).get("throughput_ratio", 1) < 0.5:
            recommendations.append({
                "priority": "medium",
                "area": "Performance",
                "recommendation": "HTTP throughput is below 50% of reference benchmarks. Profile the request handling path, consider connection pooling optimization, and review middleware overhead.",
            })

        if comparison.get("telemetry", {}).get("throughput_ratio", 1) < 0.5:
            recommendations.append({
                "priority": "medium",
                "area": "Ingestion",
                "recommendation": "Telemetry throughput is below reference. Consider batching database inserts, using bulk insert operations, or implementing a write-ahead buffer.",
            })

        return recommendations

    def generate_plots(self, output_dir: str) -> List[str]:
        """Generate visualization plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available, skipping plot generation")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        generated_plots = []

        # Plot 1: Latency distribution by endpoint
        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if http_results and "endpoints" in http_results:
            fig, ax = plt.subplots(figsize=(12, 6))

            endpoints = list(http_results["endpoints"].keys())
            p50s = [http_results["endpoints"][e].get("latency", {}).get("p50_ms", 0) for e in endpoints]
            p95s = [http_results["endpoints"][e].get("latency", {}).get("p95_ms", 0) for e in endpoints]
            p99s = [http_results["endpoints"][e].get("latency", {}).get("p99_ms", 0) for e in endpoints]

            x = range(len(endpoints))
            width = 0.25

            ax.bar([i - width for i in x], p50s, width, label='p50', color='#2ecc71')
            ax.bar(x, p95s, width, label='p95', color='#f39c12')
            ax.bar([i + width for i in x], p99s, width, label='p99', color='#e74c3c')

            ax.set_ylabel('Latency (ms)')
            ax.set_title('API Endpoint Latency Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels([e.split('/')[-1] or 'root' for e in endpoints], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plot_path = output_path / "latency_distribution.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            generated_plots.append(str(plot_path))

        # Plot 2: Throughput comparison
        comparison = self.analysis.get("comparison", {})
        if comparison:
            fig, ax = plt.subplots(figsize=(10, 6))

            categories = []
            measured = []
            reference = []

            if "http_api" in comparison:
                categories.append("HTTP API\n(RPS)")
                measured.append(comparison["http_api"]["measured"]["throughput_rps"])
                reference.append(comparison["http_api"]["reference"]["throughput_rps"])

            if "telemetry" in comparison:
                categories.append("Telemetry\n(events/s)")
                measured.append(comparison["telemetry"]["measured"]["events_per_second"])
                reference.append(comparison["telemetry"]["reference"]["throughput_rps"])

            if categories:
                x = range(len(categories))
                width = 0.35

                ax.bar([i - width/2 for i in x], measured, width, label='Measured', color='#3498db')
                ax.bar([i + width/2 for i in x], reference, width, label='Reference', color='#95a5a6')

                ax.set_ylabel('Throughput')
                ax.set_title('Throughput Comparison vs Reference Benchmarks')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plot_path = output_path / "throughput_comparison.png"
                plt.savefig(plot_path, dpi=150)
                plt.close()
                generated_plots.append(str(plot_path))

        # Plot 3: Resource utilization over time (if we have samples)
        # This would require raw samples which we don't store currently
        # Placeholder for future enhancement

        return generated_plots

    def generate_markdown_report(self, output_path: str) -> str:
        """Generate a markdown report."""
        report_lines = []

        # Header
        report_lines.append("# TensorGuardFlow Performance Benchmark Report\n")
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

        # Metadata
        metadata = self.results.get("metadata", {})
        report_lines.append("## Test Environment\n")
        report_lines.append("| Parameter | Value |")
        report_lines.append("|-----------|-------|")

        env = metadata.get("environment", {})
        report_lines.append(f"| CPU | {env.get('cpu', 'N/A')} |")
        report_lines.append(f"| Memory | {env.get('memory_gb', 'N/A')} GB |")
        report_lines.append(f"| OS | {env.get('os', 'N/A')} |")
        report_lines.append(f"| Python | {env.get('python', 'N/A')} |")

        config = metadata.get("config", {})
        load = config.get("load_level", {})
        report_lines.append(f"| Concurrent Users | {load.get('concurrent_users', 'N/A')} |")
        report_lines.append(f"| Duration | {load.get('duration_seconds', 'N/A')}s |")
        report_lines.append("")

        # Summary
        summary = self.results.get("summary", {})
        status = "PASSED" if summary.get("passed") else "FAILED"
        report_lines.append(f"## Overall Result: **{status}**\n")

        if summary.get("checks"):
            report_lines.append("### Threshold Checks\n")
            report_lines.append("| Check | Result | Value | Threshold |")
            report_lines.append("|-------|--------|-------|-----------|")

            for check in summary["checks"]:
                result = "Pass" if check["passed"] else "**FAIL**"
                report_lines.append(f"| {check['name']} | {result} | {check['value']:.2f} | {check['threshold']:.2f} |")
            report_lines.append("")

        # HTTP API Results
        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if http_results and "endpoints" in http_results:
            report_lines.append("## HTTP API Benchmarks\n")
            report_lines.append("| Endpoint | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Error Rate |")
            report_lines.append("|----------|-----|----------|----------|----------|------------|")

            for path, metrics in http_results["endpoints"].items():
                latency = metrics.get("latency", {})
                report_lines.append(
                    f"| `{path}` | "
                    f"{metrics.get('throughput', {}).get('requests_per_second', 0):.0f} | "
                    f"{latency.get('p50_ms', 0):.1f} | "
                    f"{latency.get('p95_ms', 0):.1f} | "
                    f"{latency.get('p99_ms', 0):.1f} | "
                    f"{metrics.get('error_rate', 0):.2%} |"
                )
            report_lines.append("")

        # Telemetry Results
        telemetry_results = self.results.get("benchmarks", {}).get("telemetry_ingest", {})
        if telemetry_results and "metrics" in telemetry_results:
            metrics = telemetry_results["metrics"]
            report_lines.append("## Telemetry Ingestion Benchmarks\n")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")
            report_lines.append(f"| Total Events | {metrics.get('total_events', 0):,} |")
            report_lines.append(f"| Events/Second | {metrics.get('throughput', {}).get('events_per_second', 0):,.0f} |")
            report_lines.append(f"| Batches/Second | {metrics.get('throughput', {}).get('batches_per_second', 0):,.1f} |")
            report_lines.append(f"| Acceptance Rate | {metrics.get('acceptance_rate', 0):.2%} |")
            report_lines.append(f"| p50 Latency | {metrics.get('latency', {}).get('p50_ms', 0):.1f} ms |")
            report_lines.append(f"| p95 Latency | {metrics.get('latency', {}).get('p95_ms', 0):.1f} ms |")
            report_lines.append(f"| p99 Latency | {metrics.get('latency', {}).get('p99_ms', 0):.1f} ms |")
            report_lines.append("")

        # Comparative Analysis
        comparison = self.analysis.get("comparison", {})
        if comparison:
            report_lines.append("## Comparative Analysis\n")
            report_lines.append("Comparison against published benchmarks:\n")

            if "http_api" in comparison:
                http_comp = comparison["http_api"]
                ratio = http_comp.get("throughput_ratio", 0)
                status = "within expected range" if http_comp.get("within_order_of_magnitude") else "below expected"
                report_lines.append(f"### HTTP API vs FastAPI Reference\n")
                report_lines.append(f"- Measured: {http_comp['measured']['throughput_rps']:.0f} RPS")
                report_lines.append(f"- Reference: {http_comp['reference']['throughput_rps']} RPS ({http_comp['reference']['source']})")
                report_lines.append(f"- Ratio: {ratio:.2f}x ({status})\n")

            if "telemetry" in comparison:
                tel_comp = comparison["telemetry"]
                ratio = tel_comp.get("throughput_ratio", 0)
                status = "within expected range" if tel_comp.get("within_order_of_magnitude") else "below expected"
                report_lines.append(f"### Telemetry vs OpenTelemetry Collector\n")
                report_lines.append(f"- Measured: {tel_comp['measured']['events_per_second']:.0f} events/s")
                report_lines.append(f"- Reference: {tel_comp['reference']['throughput_rps']} RPS ({tel_comp['reference']['source']})")
                report_lines.append(f"- Ratio: {ratio:.2f}x ({status})\n")

        # Bottlenecks
        bottlenecks = self.analysis.get("bottlenecks", [])
        if bottlenecks:
            report_lines.append("## Identified Bottlenecks\n")
            for bn in bottlenecks:
                severity_icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(bn.get("severity", ""), "!")
                report_lines.append(f"- [{severity_icon}] **{bn.get('type', 'unknown')}** ({bn.get('component', '')}): {bn.get('detail', '')}")
            report_lines.append("")

        # Recommendations
        recommendations = self.analysis.get("recommendations", [])
        if recommendations:
            report_lines.append("## Recommendations\n")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"### {i}. {rec.get('area', 'General')} ({rec.get('priority', 'medium')} priority)\n")
                report_lines.append(f"{rec.get('recommendation', '')}\n")

        # Footer
        report_lines.append("---\n")
        report_lines.append("*Report generated by TensorGuardFlow Benchmark Suite*\n")

        report_content = "\n".join(report_lines)

        # Save report
        with open(output_path, "w") as f:
            f.write(report_content)

        return report_content


def analyze_results(results_path: str, output_dir: str = "docs") -> Dict[str, Any]:
    """
    Analyze benchmark results and generate report.

    Args:
        results_path: Path to benchmark results JSON
        output_dir: Directory for output files

    Returns:
        Analysis results dictionary
    """
    analyzer = BenchmarkAnalyzer(results_path)
    analysis = analyzer.analyze()

    # Generate plots
    plots_dir = Path(output_dir) / "images" / "benchmarks"
    plots = analyzer.generate_plots(str(plots_dir))
    analysis["plots"] = plots

    # Generate markdown report
    report_path = Path(output_dir) / "performance_benchmark_report.md"
    analyzer.generate_markdown_report(str(report_path))
    analysis["report_path"] = str(report_path)

    print(f"Analysis complete. Report saved to: {report_path}")

    return analysis
