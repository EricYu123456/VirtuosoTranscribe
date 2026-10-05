"""Loopback-only HTTP demo with bounded score uploads and trusted file serving."""

from __future__ import annotations

from dataclasses import replace
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
import threading
from urllib.parse import unquote, urlsplit
import uuid
import zipfile

from src.baselines.rule_based import PRESETS, load_preset
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.inference.audio import TempoMap, render_audio
from src.inference.export import export_score
from src.inference.generate import generate_candidates
from src.inference.rerank import rank_candidates


CONTROL_NAMES = {"virtuosity", "melody_preservation", "octave_doubling", "register_expansion",
                 "chord_thickening", "figuration_density"}
SCORE_EXTENSIONS = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}
PUBLIC_FILES = {"best.musicxml", "best.mid", "best.wav", "best.json", "source.wav",
                "source.musicxml", "alignment.html", "alignment.json", "analysis.json", "ranking.json"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def validate_upload(filename: str, data: bytes) -> str:
    """Reject traversal names, malformed containers, entity declarations, and zip bombs."""
    if not filename or any(c in filename for c in ("/", "\\", ":", "\x00")) or filename in {".", ".."}:
        raise ValueError("Upload filename must be a simple file name")
    extension = Path(filename).suffix.lower()
    if extension not in SCORE_EXTENSIONS:
        raise ValueError("Upload MusicXML, compressed MusicXML (.mxl), or MIDI")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Score upload must contain 1 byte to 8 MiB")

    def check_xml(content: bytes):
        if b"\x00" in content:
            raise ValueError("Upload XML encoded as UTF-8 rather than UTF-16 or UTF-32")
        lowered = content.lower()
        if b"<!entity" in lowered:
            raise ValueError("XML entity declarations are not supported")
        declarations = re.findall(rb"<!DOCTYPE\b[^>]*>", content, flags=re.IGNORECASE)
        if any(b"[" in declaration or not re.match(rb"<!DOCTYPE\s+score-(partwise|timewise)\s+(PUBLIC|SYSTEM)\s", declaration,
                                                   flags=re.IGNORECASE) for declaration in declarations):
            raise ValueError("Only standard external MusicXML document types are supported")
        if b"<score-partwise" not in lowered and b"<score-timewise" not in lowered and b"<container" not in lowered:
            raise ValueError("Expected a MusicXML score or compressed MusicXML container")

    if extension in {".xml", ".musicxml"}:
        check_xml(data)
    elif extension in {".mid", ".midi"}:
        if not data.startswith(b"MThd"):
            raise ValueError("Expected a standard MIDI file")
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = archive.infolist()
                if len(members) > 64 or sum(m.file_size for m in members) > 32 * 1024 * 1024:
                    raise ValueError("Compressed score expands beyond the upload limit")
                found_score = False
                for member in members:
                    path = PurePosixPath(member.filename)
                    if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                            or ":" in member.filename or member.flag_bits & 1):
                        raise ValueError("Unsafe compressed score member")
                    if member.file_size > 16 * 1024 * 1024:
                        raise ValueError("Compressed score member is too large")
                    if not member.is_dir() and path.suffix.lower() in {".xml", ".musicxml"}:
                        content = archive.read(member)
                        check_xml(content)
                        found_score |= b"<score-" in content.lower()
                if not found_score:
                    raise ValueError("Compressed score contains no MusicXML score")
        except zipfile.BadZipFile as exc:
            raise ValueError("Invalid compressed MusicXML archive") from exc
    return extension


def _without_document_types(data: bytes, extension: str) -> bytes:
    """Remove already-validated external DTD references before handing input to music21."""
    def clean(content):
        return re.sub(rb"<!DOCTYPE\b[^>]*>", b"", content, flags=re.IGNORECASE)
    if extension in {".xml", ".musicxml"}:
        return clean(data)
    if extension == ".mxl":
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
            for member in source.infolist():
                content = source.read(member)
                if PurePosixPath(member.filename).suffix.lower() in {".xml", ".musicxml"}:
                    content = clean(content)
                target.writestr(member.filename, content)
        return output.getvalue()
    return data


def _multipart(content_type: str, body: bytes) -> tuple[str, bytes, dict[str, str]]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data with a score field")
    fields, upload = {}, None
    parts = list(message.iter_parts())
    if len(parts) > 20:
        raise ValueError("Too many upload fields")
    for part in parts:
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        data = part.get_payload(decode=True)
        if name is None or not isinstance(data, bytes):
            raise ValueError("Invalid multipart upload field")
        if filename is not None:
            if name != "score" or upload is not None:
                raise ValueError("Upload exactly one score file")
            upload = filename, data
        else:
            if name in fields or len(data) > 512:
                raise ValueError("Invalid or repeated control field")
            fields[name] = data.decode("utf-8")
    if upload is None:
        raise ValueError("Missing score file")
    return *upload, fields


def generate_job(source_path: Path, output_dir: Path, fields: dict[str, str], *,
                 checkpoints: dict[str, Path] | None = None) -> dict:
    """Shared HTTP/CLI orchestration; checkpoint choices must come from the caller's inventory."""
    checkpoints = checkpoints or {}
    allowed_fields = CONTROL_NAMES | {"preset", "checkpoint", "baseline", "num_candidates", "seed",
                                      "melody_mode", "solo_part", "accompaniment_parts"}
    unknown = set(fields) - allowed_fields
    if unknown:
        raise ValueError("Unknown request controls: " + ", ".join(sorted(unknown)))
    preset = fields.get("preset", "faithful").lower()
    if preset not in PRESETS:
        raise ValueError("Unknown arrangement preset")
    values = {}
    for name in CONTROL_NAMES:
        if name in fields:
            try:
                value = float(fields[name])
            except ValueError as exc:
                raise ValueError(f"Invalid value for {name}") from exc
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in 0..1")
            values[name] = value
    checkpoint_name = fields.get("checkpoint", "")
    baseline = fields.get("baseline", "rule_based")
    if baseline not in {"literal", "rule_based"}:
        raise ValueError("Baseline must be literal or rule_based")
    try:
        count, seed = int(fields.get("num_candidates", "4")), int(fields.get("seed", "0"))
    except ValueError as exc:
        raise ValueError("Candidate count and seed must be integers") from exc
    if not 1 <= count <= 8 or not 0 <= seed < 2 ** 31:
        raise ValueError("Choose 1..8 candidates and a seed in 0..2147483647")
    melody_mode = fields.get("melody_mode", "hard")
    if melody_mode not in {"none", "soft", "hard"}:
        raise ValueError("Melody mode must be none, soft, or hard")
    accompaniment = [p.strip() for p in fields.get("accompaniment_parts", "").split(",") if p.strip()]
    source = normalize_source(parse_score(source_path), solo_part=fields.get("solo_part") or None,
                              accompaniment_parts=accompaniment or None)
    if len(source.notes) > 2500 or len(source.score.measures) > 128 or source.duration > 1024:
        raise ValueError("Interactive preview supports at most 2500 notes, 128 measures, and 1024 quarter notes; use CLI for longer scores")
    if TempoMap(source.score).seconds_at(source.duration) + .25 > 600:
        raise ValueError("Interactive audio preview supports at most 600 seconds; use a shorter excerpt")
    warnings = list(source.score.warnings)
    if checkpoint_name:
        if checkpoint_name not in checkpoints:
            raise ValueError("Checkpoint is not in the configured trusted inventory")
        from src.inference.neural_generate import generate_neural_candidates
        controls = dict(load_preset(preset).to_dict(), **values)
        candidates = generate_neural_candidates(source, checkpoints[checkpoint_name], controls,
                                               num_candidates=count, seed=seed, melody_mode=melody_mode)
        ranking_controls = controls
        generator = f"checkpoint:{checkpoint_name}"
    else:
        if values.get("melody_preservation", 1) != 1:
            warnings.append("Rule-based generation retains the source melody strictly; melody preservation remains 1. Use a checkpoint for soft/flexible modes.")
        values["melody_preservation"] = 1.0
        ranking_controls = replace(load_preset(preset), **values)
        controls = ranking_controls.to_dict()
        candidates = generate_candidates(source, ranking_controls, num_candidates=count, seed=seed,
                                         baseline=baseline)
        generator = baseline
    ranking = rank_candidates(source, candidates, ranking_controls)
    best = candidates[ranking[0]["candidate_index"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, score in (("best", best), ("source", source.score)):
        export_score(score, output_dir / f"{name}.musicxml")
    export_score(best, output_dir / "best.mid")
    best.save(output_dir / "best.json")
    audio = render_audio(best, output_dir / "best.wav")
    render_audio(source.score, output_dir / "source.wav")
    from src.analysis.melody_alignment import align_solo_to_piano
    from src.analysis.visualization import write_alignment_visualization
    alignment = align_solo_to_piano(source.solo, best, source_score=source.score)
    alignment.save(output_dir / "alignment.json")
    write_alignment_visualization(source.solo, best, alignment, output_dir / "alignment.html",
                                  title=f"{source.score.title}: solo → piano")
    metrics = ranking[0]["metrics"]
    payload = {"generator": generator, "neural_model": bool(checkpoint_name), "preset": preset,
               "controls": controls, "metrics": metrics, "audio": audio,
               "requested_candidates": count, "ranking": ranking,
               "warnings": warnings + list(best.warnings)}
    for name, obj in (("analysis.json", metrics), ("ranking.json", payload)):
        (output_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def create_server(output_root: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
                  checkpoint_root: str | Path | None = None) -> ThreadingHTTPServer:
    """Create, but do not start, a loopback server. Port 0 supports isolated tests."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The demo binds only to loopback; choose 127.0.0.1 or localhost")
    if host == "::1":
        raise ValueError("Use 127.0.0.1 for this IPv4 loopback server")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    if checkpoint_root:
        trusted = Path(checkpoint_root).resolve()
        if not trusted.is_dir():
            raise ValueError("Configured checkpoint root does not exist")
        for path in sorted(trusted.rglob("*.pt")):
            if path.is_file() and path.resolve().is_relative_to(trusted):
                checkpoints[path.relative_to(trusted).as_posix()] = path.resolve()
    gate = threading.BoundedSemaphore(1)
    ui = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "VirtuosoTranscribe/1"

        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def respond(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, payload: dict):
            self.respond(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def trusted_request(self) -> bool:
            port_number = self.server.server_address[1]
            allowed_hosts = {f"127.0.0.1:{port_number}", f"localhost:{port_number}"}
            if self.headers.get("Host") not in allowed_hosts:
                self.json_response(403, {"error": "Untrusted Host header"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in allowed_hosts}:
                self.json_response(403, {"error": "Cross-origin requests are not accepted"})
                return False
            return True

        def do_GET(self):
            if not self.trusted_request():
                return
            request_path = unquote(urlsplit(self.path).path)
            if request_path == "/":
                self.respond(200, ui, "text/html; charset=utf-8")
                return
            if request_path == "/api/health":
                self.json_response(200, {"status": "ok", "checkpoints": list(checkpoints),
                                         "max_upload_bytes": MAX_UPLOAD_BYTES, "audio": "synthetic"})
                return
            pieces = request_path.split("/")
            if len(pieces) != 4 or pieces[1] != "files" or pieces[3] not in PUBLIC_FILES:
                self.json_response(404, {"error": "File not found"})
                return
            try:
                if uuid.UUID(pieces[2]).hex != pieces[2]:
                    raise ValueError("Invalid job identifier")
                path = (root / pieces[2] / pieces[3]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("File not found")
            except ValueError:
                self.json_response(404, {"error": "File not found"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.respond(200, path.read_bytes(), content_type)

        def do_POST(self):
            if not self.trusted_request():
                self.close_connection = True
                return
            if urlsplit(self.path).path != "/api/generate":
                self.json_response(404, {"error": "Endpoint not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= MAX_UPLOAD_BYTES + 32 * 1024:
                    raise ValueError("Request body exceeds the 8 MiB score upload limit")
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data;"):
                    raise ValueError("Expected multipart/form-data")
                if not gate.acquire(blocking=False):
                    self.json_response(429, {"error": "Another generation is running; try again when it finishes"})
                    return
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("Incomplete upload")
                    filename, data, fields = _multipart(content_type, body)
                    extension = validate_upload(filename, data)
                    job = uuid.uuid4().hex
                    directory = root / job
                    directory.mkdir()
                    source_path = directory / f"upload{extension}"
                    source_path.write_bytes(_without_document_types(data, extension))
                    result = generate_job(source_path, directory, fields, checkpoints=checkpoints)
                    result["job_id"] = job
                    result["files"] = {name: f"/files/{job}/{name}" for name in PUBLIC_FILES
                                       if (directory / name).is_file()}
                    self.json_response(200, result)
                finally:
                    gate.release()
            except (ValueError, OSError, UnicodeError) as exc:
                self.close_connection = True
                self.json_response(400, {"error": str(exc)[:1000]})
            except Exception as exc:
                # Keep malformed model inputs and runtime failures from killing
                # the server; return an explicit failure, never substitute a baseline.
                self.close_connection = True
                self.json_response(500, {"error": f"Generation failed ({type(exc).__name__}): {str(exc)[:800]}"})

        def log_message(self, format, *args):
            # Keep local console logs useful without echoing uploaded score paths.
            if args:
                super().log_message(format, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
