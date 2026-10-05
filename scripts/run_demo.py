"""Launch the local interactive MusicXML/MIDI transcription demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from src.demo.server import create_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/interactive"))
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--open", action="store_true", help="Open the local demo in the default browser")
    args = parser.parse_args(argv)
    try:
        server = create_server(args.output_dir, port=args.port, checkpoint_root=args.checkpoint_root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"VirtuosoTranscribe local demo: {url}")
    print(f"Outputs: {args.output_dir.resolve()}; stop with Ctrl+C")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
