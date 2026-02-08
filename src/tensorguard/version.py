"""
Single source of truth for the TenSafe package version.

In a wheel install, the version comes from importlib.metadata (set by
pyproject.toml).  During editable / dev installs the fallback is the
hardcoded _FALLBACK string.
"""

_FALLBACK = "4.1.0"


def tensafe_version() -> str:
    """Return the installed package version, or a dev fallback."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version("tensafe")
    except Exception:
        return _FALLBACK
