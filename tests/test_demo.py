import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import pytest

from scripts.make_demo_data import make_demo_data
from src.demo.server import create_server, validate_upload


@pytest.fixture
def live_demo(tmp_path):
    server = create_server(tmp_path / "jobs", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def multipart(filename, content, fields=None):
    boundary = "test_boundary_virtuoso"
    chunks = []
    for name, value in (fields or {}).items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="score"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + content + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def test_demo_post_produces_listening_score_metrics_and_downloads(live_demo):
    url, tmp_path = live_demo
    make_demo_data(tmp_path / "source")
    body, content_type = multipart("study.musicxml", (tmp_path / "source/source.musicxml").read_bytes(),
                                   {"preset": "romantic", "num_candidates": "2", "octave_doubling": "1"})
    with urlopen(Request(url + "/api/generate", body, {"Content-Type": content_type}), timeout=30) as response:
        result = json.load(response)
    assert result["generator"] == "rule_based"
    assert result["metrics"]["melody_coverage_octave_tolerant"] == 1
    assert result["audio"]["engine"] == "synthetic"
    for file in ("best.musicxml", "best.mid", "best.wav", "analysis.json", "alignment.html"):
        with urlopen(url + result["files"][file]) as response:
            assert len(response.read()) > 10
    with urlopen(url + "/api/health") as response:
        assert json.load(response)["status"] == "ok"
    with pytest.raises(HTTPError) as error:
        urlopen(url + f"/files/{result['job_id']}/%2e%2e%2fsource.musicxml")
    assert error.value.code == 404


def test_invalid_upload_and_cross_origin_requests_are_explicit(live_demo):
    url, _ = live_demo
    body, content_type = multipart("../../score.musicxml", b"<score-partwise/>")
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/api/generate", body, {"Content-Type": content_type}))
    assert error.value.code == 400
    assert "filename" in json.load(error.value)["error"]
    body, content_type = multipart("score.musicxml", b"<score-partwise/>")
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/api/generate", body, {"Content-Type": content_type, "Origin": "https://untrusted.example"}))
    assert error.value.code == 403


def test_upload_rejects_entity_expansion_archive_traversal_and_invalid_midi():
    with pytest.raises(ValueError, match="entity"):
        validate_upload("score.musicxml", b'<!DOCTYPE score-partwise [<!ENTITY boom "boom">]><score-partwise/>')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../score.musicxml", b"<score-partwise/>")
    with pytest.raises(ValueError, match="Unsafe"):
        validate_upload("score.mxl", archive.getvalue())
    with pytest.raises(ValueError, match="MIDI"):
        validate_upload("score.mid", b"not midi")
    with pytest.raises(ValueError, match="loopback"):
        create_server("unused", host="0.0.0.0")


def test_demo_accepts_compressed_musicxml_and_releases_slot_after_a_parse_error(live_demo):
    url, tmp_path = live_demo
    invalid, content_type = multipart("bad.musicxml", b"<score-partwise/>")
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/api/generate", invalid, {"Content-Type": content_type}))
    assert error.value.code == 400
    make_demo_data(tmp_path / "compressed")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="score.musicxml"/></rootfiles></container>')
        writer.writestr("score.musicxml", (tmp_path / "compressed/source.musicxml").read_bytes())
    body, content_type = multipart("score.mxl", archive.getvalue(), {"num_candidates": "1"})
    with urlopen(Request(url + "/api/generate", body, {"Content-Type": content_type}), timeout=30) as response:
        result = json.load(response)
    assert result["metrics"]["melody_coverage_octave_tolerant"] == 1
