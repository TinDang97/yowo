"""A credentialed weights URL is not echoed, logged or raised.

Red-first for ADD task `weights-url-redaction`.

`redact_url()` already exists (built by `rtsp-credential-redaction`) and is applied
here, not reimplemented. Three emit points interpolate a URL and must redact it:

    cli/_main.py::models_command      -- prints m.default_weights_url per row
    models/_weights.py::_download            -- retries-exhausted message
    models/_weights.py::_attempt_download    -- stalled-download message

`_warn_unpinned` names a weight stem, never a URL, and is verified clean rather
than changed.
"""

from __future__ import annotations

from unittest import mock

import pytest

from yowo.errors import ModelNotFoundError
from yowo.models._registry import ModelMeta, get, register
from yowo.models._weights import _attempt_download, _download
from yowo.types import ModelFamily, ModelSize

USER = "AKIAABCDEFGHIJKLMNOP"
PASSWORD = "s3cr3t-signing-key"
HOST = "private-bucket.s3.amazonaws.com"
PATH = "/weights/custom_n.pt"
CREDENTIALED_URL = f"https://{USER}:{PASSWORD}@{HOST}{PATH}"

# A malformed authority: the "port" segment cannot parse as an int, so
# redact_url() cannot decompose it and must yield the literal sentinel
# rather than fall back to the raw (still-credentialed) string.
UNPARSEABLE_URL = f"https://{USER}:{PASSWORD}@host:notaport{PATH}"


def _no_credential(text: str) -> None:
    assert PASSWORD not in text, f"password leaked: {text!r}"
    assert USER not in text, f"username leaked (half a credential): {text!r}"


@pytest.fixture
def custom_model_registered():
    """Register a credentialed custom model, restoring the registry after."""
    from yowo.models import _registry as reg

    key = (ModelFamily.YOLO11, ModelSize.NANO)
    saved = reg._REGISTRY.get(key)

    meta = ModelMeta(
        family=ModelFamily.YOLO11,
        size=ModelSize.NANO,
        input_height=640,
        input_width=640,
        num_classes=10,
        weight_stem="custom_n",
        default_weights_url=CREDENTIALED_URL,
    )
    register(meta)
    try:
        yield meta
    finally:
        if saved is not None:
            reg._REGISTRY[key] = saved
        else:
            reg._REGISTRY.pop(key, None)


class TestModelsCommandRedaction:
    def test_models_command_never_prints_userinfo(self, custom_model_registered) -> None:
        """covers: M1, R:USERINFO, A2 -- a registered credentialed URL never reaches stdout."""
        from click.testing import CliRunner

        from yowo.cli._main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        _no_credential(result.output)

    def test_models_command_leaves_clean_urls_byte_identical(self) -> None:
        """covers: M2, A5, E1 -- the 10 built-in (credential-free) URLs are unchanged,
        and the table keeps its four columns and widths."""
        from click.testing import CliRunner

        from yowo.cli._main import cli
        from yowo.models import list_available

        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0

        header = f"{'Model':<12} {'Input':<10} {'Classes':<10} {'URL'}"
        assert result.output.splitlines()[0] == header

        for m in list_available():
            name = f"{m.family.value}{m.size.value}"
            expected_row = (
                f"{name:<12} {m.input_height}x{m.input_width:<5} "
                f"{m.num_classes:<10} {m.default_weights_url}"
            )
            assert expected_row in result.output, f"row for {name} was altered: {expected_row!r}"


class TestDownloadDiagnosticsRedaction:
    def test_download_failure_message_redacts_the_url(self, tmp_path) -> None:
        """covers: M1, A2, E3 -- retries exhausted; host and path still present."""
        dest = tmp_path / "custom_n.pt"
        with (
            mock.patch("yowo.models._weights.requests.get", side_effect=ConnectionError("refused")),
            mock.patch("yowo.models._weights.time.sleep"),
            pytest.raises(ModelNotFoundError) as excinfo,
        ):
            _download(CREDENTIALED_URL, dest)

        message = str(excinfo.value)
        _no_credential(message)
        assert HOST in message
        assert PATH in message

    def test_stalled_download_message_redacts_the_url(self, tmp_path) -> None:
        """covers: M1, A2 -- the chunk-deadline path."""
        tmp_dest = tmp_path / "custom_n.pt.tmp"

        fake_response = mock.Mock()
        fake_response.raise_for_status.return_value = None
        fake_response.headers = {}
        fake_response.iter_content.return_value = [b"\x00" * 8192]

        # First call establishes the deadline (t=0); the second call, made
        # inside the chunk loop, is already past it -- the loop must raise
        # before ever writing or requesting another chunk.
        with (
            mock.patch("yowo.models._weights.requests.get", return_value=fake_response),
            mock.patch("yowo.models._weights.time.monotonic", side_effect=[0.0, 301.0]),
            pytest.raises(ModelNotFoundError) as excinfo,
        ):
            _attempt_download(CREDENTIALED_URL, tmp_dest, suppress_progress=True)

        message = str(excinfo.value)
        _no_credential(message)
        assert HOST in message
        assert PATH in message

    def test_unparseable_url_never_falls_back_to_raw(self, tmp_path) -> None:
        """covers: R:RAWFALLBACK, A4, E2 -- a malformed authority emits the sentinel,
        never the raw (still-credentialed) string."""
        dest = tmp_path / "custom_n.pt"
        with (
            mock.patch("yowo.models._weights.requests.get", side_effect=ConnectionError("refused")),
            mock.patch("yowo.models._weights.time.sleep"),
            pytest.raises(ModelNotFoundError) as excinfo,
        ):
            _download(UNPARSEABLE_URL, dest)

        message = str(excinfo.value)
        _no_credential(message)
        assert "<unparseable url>" in message
        assert UNPARSEABLE_URL not in message


class TestRegistryUnaffected:
    def test_registry_still_stores_the_credential(self, custom_model_registered) -> None:
        """covers: M3 -- redaction happens at the emit point, not at registration;
        the registry keeps the fetchable, credentialed URL for `_download` to use."""
        retrieved = get(ModelFamily.YOLO11, ModelSize.NANO)
        assert retrieved.default_weights_url == CREDENTIALED_URL


class TestWarnUnpinnedVerifiedClean:
    def test_warn_unpinned_never_interpolates_a_url(self) -> None:
        """covers: A2 -- `_warn_unpinned` names a weight stem only, no URL; it is
        verified clean rather than changed by this task."""
        import inspect

        from yowo.models import _weights as weights_mod

        body = inspect.getsource(weights_mod._warn_unpinned)
        assert "url" not in body.lower()


class TestReadmeExample:
    def test_readme_example_registers_no_credentialed_url(self) -> None:
        """covers: M4 -- the registration example carries no credential, and the
        doc states that a credential in default_weights_url is redacted on display."""
        import re
        from pathlib import Path

        readme = Path(__file__).parents[2] / "src" / "yowo" / "models" / "README.md"
        text = readme.read_text(encoding="utf-8")

        url_lines = [line for line in text.splitlines() if "default_weights_url" in line]
        assert url_lines, "expected to find the registration example's default_weights_url= line"
        credentialed = re.compile(r"://[^/\s\"']*@")
        for line in url_lines:
            assert not credentialed.search(line), (
                f"the registration example must not model a credentialed URL: {line!r}"
            )

        assert "redact" in text.lower(), (
            "the README must state that a credential in default_weights_url is redacted "
            "wherever it is displayed"
        )
