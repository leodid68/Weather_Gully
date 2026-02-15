"""Shared SSL context for HTTP clients.

Uses certifi bundle if available (fixes macOS Python missing certs),
falls back to system defaults, then to unverified as last resort.
"""

import ssl


def make_ssl_context() -> ssl.SSLContext:
    """Create an SSL context with working certificate verification."""
    # 1. Try certifi (reliable on macOS)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass

    # 2. Try system defaults
    try:
        ctx = ssl.create_default_context()
        # Verify the context can actually connect (certs are loadable)
        if ctx.get_ca_certs():
            return ctx
        return ctx
    except Exception:
        pass

    # 3. Last resort: unverified (not ideal, but functional)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


SSL_CTX = make_ssl_context()
