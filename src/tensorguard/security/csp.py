"""
Content Security Policy (CSP) Module.

Provides comprehensive CSP header management:
- Strict CSP directives for XSS prevention
- Nonce-based script execution
- Report-only mode for testing
- Policy violation reporting
"""

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class ContentSecurityPolicy:
    """
    Content Security Policy configuration.

    Implements a strict CSP that prevents:
    - Cross-site scripting (XSS)
    - Data injection attacks
    - Clickjacking (via frame-ancestors)
    - Mixed content
    """

    # Default sources
    default_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Script sources
    script_src: List[str] = field(default_factory=lambda: ["'self'"])
    script_src_elem: Optional[List[str]] = None
    script_src_attr: Optional[List[str]] = None

    # Style sources
    style_src: List[str] = field(default_factory=lambda: ["'self'"])
    style_src_elem: Optional[List[str]] = None
    style_src_attr: Optional[List[str]] = None

    # Image sources
    img_src: List[str] = field(default_factory=lambda: ["'self'", "data:"])

    # Font sources
    font_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Connect sources (XHR, WebSocket, etc.)
    connect_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Media sources
    media_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Object sources (plugins, etc.)
    object_src: List[str] = field(default_factory=lambda: ["'none'"])

    # Frame sources
    frame_src: List[str] = field(default_factory=lambda: ["'none'"])
    frame_ancestors: List[str] = field(default_factory=lambda: ["'none'"])
    child_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Worker sources
    worker_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Form action
    form_action: List[str] = field(default_factory=lambda: ["'self'"])

    # Base URI
    base_uri: List[str] = field(default_factory=lambda: ["'self'"])

    # Manifest source
    manifest_src: List[str] = field(default_factory=lambda: ["'self'"])

    # Upgrade insecure requests
    upgrade_insecure_requests: bool = True

    # Block mixed content
    block_all_mixed_content: bool = True

    # Reporting
    report_uri: Optional[str] = None
    report_to: Optional[str] = None

    # Nonce configuration
    use_nonces: bool = True
    nonce_length: int = 32

    # Report-only mode (for testing)
    report_only: bool = False

    @classmethod
    def strict(cls) -> "ContentSecurityPolicy":
        """Create a strict CSP suitable for APIs."""
        return cls(
            default_src=["'none'"],
            script_src=["'none'"],
            style_src=["'none'"],
            img_src=["'none'"],
            font_src=["'none'"],
            connect_src=["'self'"],
            media_src=["'none'"],
            object_src=["'none'"],
            frame_src=["'none'"],
            frame_ancestors=["'none'"],
            child_src=["'none'"],
            worker_src=["'none'"],
            form_action=["'self'"],
            base_uri=["'none'"],
            manifest_src=["'none'"],
            upgrade_insecure_requests=True,
            block_all_mixed_content=True,
            use_nonces=False,
        )

    @classmethod
    def for_api(cls) -> "ContentSecurityPolicy":
        """Create CSP optimized for API-only applications."""
        return cls(
            default_src=["'none'"],
            script_src=["'none'"],
            style_src=["'none'"],
            img_src=["'none'"],
            font_src=["'none'"],
            connect_src=["'self'"],
            media_src=["'none'"],
            object_src=["'none'"],
            frame_src=["'none'"],
            frame_ancestors=["'none'"],
            child_src=["'none'"],
            worker_src=["'none'"],
            form_action=["'none'"],
            base_uri=["'none'"],
            manifest_src=["'none'"],
            upgrade_insecure_requests=True,
            block_all_mixed_content=True,
            use_nonces=False,
        )

    @classmethod
    def for_web_ui(
        cls,
        allowed_origins: Optional[List[str]] = None,
        allow_inline_styles: bool = False,
    ) -> "ContentSecurityPolicy":
        """Create CSP for web UI applications."""
        origins = allowed_origins or ["'self'"]

        style_src = list(origins)
        if allow_inline_styles:
            style_src.append("'unsafe-inline'")

        return cls(
            default_src=["'self'"],
            script_src=list(origins),  # Will use nonces
            style_src=style_src,
            img_src=list(origins) + ["data:", "blob:"],
            font_src=list(origins) + ["data:"],
            connect_src=list(origins) + ["wss:"],  # WebSocket support
            media_src=list(origins),
            object_src=["'none'"],
            frame_src=["'none'"],
            frame_ancestors=["'self'"],
            child_src=["'self'", "blob:"],
            worker_src=["'self'", "blob:"],
            form_action=list(origins),
            base_uri=["'self'"],
            manifest_src=["'self'"],
            upgrade_insecure_requests=True,
            block_all_mixed_content=True,
            use_nonces=True,
        )

    def generate_nonce(self) -> str:
        """Generate a cryptographic nonce for scripts/styles."""
        return secrets.token_urlsafe(self.nonce_length)

    def add_hash(self, content: str, algorithm: str = "sha256") -> str:
        """
        Generate a hash for inline content.

        Args:
            content: Script or style content
            algorithm: Hash algorithm (sha256, sha384, sha512)

        Returns:
            CSP hash directive value
        """
        if algorithm == "sha256":
            digest = hashlib.sha256(content.encode()).digest()
        elif algorithm == "sha384":
            digest = hashlib.sha384(content.encode()).digest()
        elif algorithm == "sha512":
            digest = hashlib.sha512(content.encode()).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        import base64

        b64_hash = base64.b64encode(digest).decode()
        return f"'{algorithm}-{b64_hash}'"

    def to_header_value(self, nonce: Optional[str] = None) -> str:
        """
        Generate the CSP header value.

        Args:
            nonce: Optional nonce to include in script-src

        Returns:
            CSP header string
        """
        directives = []

        # Build directives
        def add_directive(name: str, values: Optional[List[str]]) -> None:
            if values:
                directive_values = list(values)

                # Add nonce to script-src and style-src if enabled
                if nonce and name in ("script-src", "style-src", "script-src-elem"):
                    directive_values.append(f"'nonce-{nonce}'")

                directives.append(f"{name} {' '.join(directive_values)}")

        add_directive("default-src", self.default_src)
        add_directive("script-src", self.script_src)
        add_directive("script-src-elem", self.script_src_elem)
        add_directive("script-src-attr", self.script_src_attr)
        add_directive("style-src", self.style_src)
        add_directive("style-src-elem", self.style_src_elem)
        add_directive("style-src-attr", self.style_src_attr)
        add_directive("img-src", self.img_src)
        add_directive("font-src", self.font_src)
        add_directive("connect-src", self.connect_src)
        add_directive("media-src", self.media_src)
        add_directive("object-src", self.object_src)
        add_directive("frame-src", self.frame_src)
        add_directive("frame-ancestors", self.frame_ancestors)
        add_directive("child-src", self.child_src)
        add_directive("worker-src", self.worker_src)
        add_directive("form-action", self.form_action)
        add_directive("base-uri", self.base_uri)
        add_directive("manifest-src", self.manifest_src)

        if self.upgrade_insecure_requests:
            directives.append("upgrade-insecure-requests")

        if self.block_all_mixed_content:
            directives.append("block-all-mixed-content")

        if self.report_uri:
            directives.append(f"report-uri {self.report_uri}")

        if self.report_to:
            directives.append(f"report-to {self.report_to}")

        return "; ".join(directives)

    def get_header_name(self) -> str:
        """Get the appropriate header name based on mode."""
        if self.report_only:
            return "Content-Security-Policy-Report-Only"
        return "Content-Security-Policy"


class CSPMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for Content Security Policy headers.

    Automatically adds CSP headers to all responses with
    optional nonce generation for scripts.
    """

    def __init__(
        self,
        app,
        policy: Optional[ContentSecurityPolicy] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        """
        Initialize CSP middleware.

        Args:
            app: FastAPI application
            policy: Content Security Policy configuration
            exclude_paths: Paths to exclude from CSP headers
        """
        super().__init__(app)
        self.policy = policy or ContentSecurityPolicy.for_api()
        self.exclude_paths = set(exclude_paths or [])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add CSP headers to response."""
        # Check exclusions
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate nonce if needed
        nonce = None
        if self.policy.use_nonces:
            nonce = self.policy.generate_nonce()
            # Store nonce in request state for template use
            request.state.csp_nonce = nonce

        # Process request
        response = await call_next(request)

        # Add CSP header
        header_name = self.policy.get_header_name()
        header_value = self.policy.to_header_value(nonce)
        response.headers[header_name] = header_value

        # Add nonce header for client-side use if needed
        if nonce:
            response.headers["X-CSP-Nonce"] = nonce

        return response


class CSPViolationHandler:
    """
    Handler for CSP violation reports.

    Processes and logs CSP violations for security monitoring.
    """

    def __init__(self, log_violations: bool = True):
        """
        Initialize violation handler.

        Args:
            log_violations: Whether to log violations
        """
        self.log_violations = log_violations
        self._violation_counts: Dict[str, int] = {}

    async def handle_report(self, report: Dict[str, Any]) -> None:
        """
        Handle a CSP violation report.

        Args:
            report: CSP violation report
        """
        csp_report = report.get("csp-report", report)

        violated_directive = csp_report.get("violated-directive", "unknown")
        blocked_uri = csp_report.get("blocked-uri", "unknown")
        document_uri = csp_report.get("document-uri", "unknown")
        source_file = csp_report.get("source-file", "")
        line_number = csp_report.get("line-number", 0)

        # Track violation counts
        key = f"{violated_directive}:{blocked_uri}"
        self._violation_counts[key] = self._violation_counts.get(key, 0) + 1

        if self.log_violations:
            logger.warning(
                f"CSP Violation: {violated_directive} blocked {blocked_uri}",
                extra={
                    "csp_violation": {
                        "directive": violated_directive,
                        "blocked_uri": blocked_uri,
                        "document_uri": document_uri,
                        "source_file": source_file,
                        "line_number": line_number,
                    }
                },
            )

    def get_violation_stats(self) -> Dict[str, int]:
        """Get violation statistics."""
        return dict(self._violation_counts)


def create_csp_report_endpoint(handler: Optional[CSPViolationHandler] = None):
    """
    Create a FastAPI endpoint for CSP violation reports.

    Usage:
        from fastapi import FastAPI
        from tensorguard.security.csp import create_csp_report_endpoint

        app = FastAPI()
        app.post("/csp-report")(create_csp_report_endpoint())
    """
    violation_handler = handler or CSPViolationHandler()

    async def csp_report_endpoint(request: Request) -> Dict[str, str]:
        """Receive CSP violation reports."""
        try:
            report = await request.json()
            await violation_handler.handle_report(report)
        except Exception as e:
            logger.warning(f"Failed to process CSP report: {e}")

        return {"status": "received"}

    return csp_report_endpoint
