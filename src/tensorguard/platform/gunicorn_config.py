"""
Gunicorn configuration for TenSafe production deployment.

This config enables multi-worker mode with uvicorn workers for async support.
Environment variables can override defaults for Kubernetes deployment.
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv("WORKERS", min(multiprocessing.cpu_count() * 2 + 1, 8)))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = int(os.getenv("TIMEOUT", "120"))
keepalive = int(os.getenv("KEEPALIVE", "5"))
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))

# Worker lifecycle (memory leak protection)
max_requests = int(os.getenv("MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "1000"))

# Process naming
proc_name = "tensafe-server"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("TG_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Preload app for better memory utilization (shared memory between workers)
preload_app = os.getenv("PRELOAD_APP", "true").lower() == "true"


def on_starting(server):
    """Called just before the master process is initialized."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info(f"TenSafe server starting with {workers} workers")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker {worker.pid} spawned")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    import logging

    logger = logging.getLogger("gunicorn.error")
    logger.info(f"Worker {worker.pid} exited")


def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug(f"Processing {req.method} {req.path}")


def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass
