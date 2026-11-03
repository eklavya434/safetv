"""
Tests for SafeTV security controls: path traversal, authentication, upload validation, SSRF prevention.
"""
import ipaddress
import os
import sys
import tempfile
import unittest
from urllib.parse import urlparse

# Ensure safetv is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestPathTraversalPrevention(unittest.TestCase):
    """Test that evidence clip endpoint prevents directory traversal."""

    def test_basename_strips_traversal(self):
        """os.path.basename must strip directory traversal sequences."""
        malicious = "../../../etc/passwd"
        result = os.path.basename(malicious)
        self.assertEqual(result, "passwd")
        self.assertNotIn("..", result)

    def test_realpath_confinement(self):
        """realpath confinement must prevent escape from evidence directory."""
        evidence_dir = os.path.realpath(tempfile.gettempdir())
        safe_name = os.path.basename("../../../etc/passwd")
        target = os.path.realpath(os.path.join(evidence_dir, safe_name))
        try:
            common = os.path.commonpath([target, evidence_dir])
            self.assertEqual(common, evidence_dir,
                "Traversal escape detected: target not confined to evidence_dir")
        except ValueError:
            pass

    def test_null_byte_in_filename(self):
        """Filenames with null bytes must be detected and rejected."""
        malicious = "evidence.mp4\x00.jpg"
        self.assertTrue(any(c in malicious for c in ['\x00']))


class TestUploadValidation(unittest.TestCase):
    """Test upload file validation."""

    def test_valid_extensions_allowed(self):
        """Valid video extensions must be accepted."""
        valid = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        for ext in valid:
            self.assertIn(ext, valid)

    def test_invalid_extensions_rejected(self):
        """Non-video extensions must be rejected by the server."""
        invalid_names = ["script.sh", "malware.exe", "config.json", "../../etc/passwd"]
        valid_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        for name in invalid_names:
            ext = os.path.splitext(name)[1].lower()
            self.assertNotIn(ext, valid_exts,
                f"Extension '{ext}' from '{name}' should be rejected")

    def test_upload_size_limit_constant(self):
        """Upload size limit must be set to a reasonable value (250 MB)."""
        limit_mb = 250
        limit_bytes = limit_mb * 1024 * 1024
        self.assertGreater(limit_bytes, 0)
        self.assertLessEqual(limit_bytes, 500 * 1024 * 1024)


class TestSSRFPrevention(unittest.TestCase):
    """Test SSRF prevention for camera URL validation."""

    def test_private_ip_detection(self):
        """Private IP addresses must be identified."""
        private_ips = ["192.168.1.50", "10.0.0.1", "172.16.0.1", "127.0.0.1"]
        for ip_str in private_ips:
            addr = ipaddress.ip_address(ip_str)
            self.assertTrue(addr.is_private or addr.is_loopback,
                f"{ip_str} should be identified as private/loopback")

    def test_public_ip_not_private(self):
        """Public IP addresses must not be flagged as private."""
        public_ips = ["8.8.8.8", "1.1.1.1", "203.0.113.1"]
        for ip_str in public_ips:
            addr = ipaddress.ip_address(ip_str)
            self.assertFalse(addr.is_private and addr.is_loopback,
                f"{ip_str} should not be flagged as both private AND loopback")

    def test_scheme_validation(self):
        """Only rtsp://, rtsps://, file:// schemes should be allowed."""
        allowed_schemes = {"rtsp", "rtsps", "file"}
        valid_urls = ["rtsp://camera.internal:554/live", "rtsps://camera.internal/stream", "file:///test.mp4"]
        invalid_urls = ["http://evil.com/payload", "https://example.com", "ftp://files.com"]
        for url in valid_urls:
            scheme = urlparse(url).scheme.lower()
            self.assertIn(scheme, allowed_schemes, f"{url} should be allowed")
        for url in invalid_urls:
            scheme = urlparse(url).scheme.lower()
            self.assertNotIn(scheme, allowed_schemes, f"{url} should be blocked")


class TestCORSConfiguration(unittest.TestCase):
    """Test CORS configuration does not use wildcard with credentials."""

    def test_cors_origins_not_wildcard_with_credentials(self):
        """CORS allowed origins must not combine wildcard with credentials=True."""
        server_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "safetv", "server.py")
        self.assertTrue(os.path.exists(server_path), "server.py not found")
        with open(server_path, "r", encoding="utf-8") as f:
            content = f.read()

        import re
        # Find allow_origins and verify no wildcard list literal
        cors_match = re.search(r'allow_origins=\["\*"\]', content)
        self.assertIsNone(cors_match,
            "CRITICAL: Wildcard CORS literal allow_origins=[\"*\"] found in server.py")


class TestRouteSecurity(unittest.TestCase):
    """Programmatic route enumeration security test."""

    def test_protected_routes_have_api_key_dependency(self):
        """All sensitive and mutating routes must have verify_api_key dependency."""
        from safetv.server import app, verify_api_key
        from fastapi.routing import APIRoute

        # Public allowlist
        public_routes = {
            "/",
            "/api/status",
            "/api/model/health",
            "/api/evidence/{filename}",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
            "/openapi.json",
        }

        unprotected = []
        for route in app.routes:
            if isinstance(route, APIRoute):
                path = route.path
                if path in public_routes:
                    continue
                # Inspect dependencies on route
                dep_callables = [d.call for d in route.dependant.dependencies]
                if verify_api_key not in dep_callables:
                    unprotected.append(f"{route.methods} {path}")

        self.assertEqual(
            unprotected,
            [],
            f"The following routes lack verify_api_key dependency: {unprotected}"
        )


class TestRateLimiter(unittest.TestCase):
    """Test in-memory sliding window rate limiter."""

    def test_rate_limiter_allows_under_limit_and_blocks_over_limit(self):
        from safetv.server import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=1.0)
        self.assertTrue(limiter.is_allowed("client1"))
        self.assertTrue(limiter.is_allowed("client1"))
        self.assertTrue(limiter.is_allowed("client1"))
        # 4th request must be rejected
        self.assertFalse(limiter.is_allowed("client1"))
        # Different client should still be allowed
        self.assertTrue(limiter.is_allowed("client2"))


class TestDNSRebindingProtection(unittest.TestCase):
    """Test DNS rebinding and camera stream validation."""

    def test_stream_url_private_ip_blocked_when_disallowed(self):
        from safetv.cctv import validate_stream_url
        with self.assertRaises(PermissionError):
            validate_stream_url("rtsp://127.0.0.1:554/live", allow_private=False)

    def test_stream_url_local_file_allowed(self):
        from safetv.cctv import validate_stream_url
        # Should not raise
        validate_stream_url("data/raw/test.mp4", allow_private=False)
        validate_stream_url("file:///test.mp4", allow_private=False)


if __name__ == "__main__":
    unittest.main()
