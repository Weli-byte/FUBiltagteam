"""Start the mock API: ``uv run python scripts/mock_server.py [port] [seed]``."""

import sys

from onkos.contracts.mock_server import make_server


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    server = make_server("127.0.0.1", port, seed)
    print(f"ONKOS mock API on http://127.0.0.1:{port} (seed={seed}); Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
