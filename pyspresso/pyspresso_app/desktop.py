"""Standalone desktop entry point.

Runs the Flask backend with a production-grade WSGI server (Waitress) and
opens the app in the user's default browser. This is the module PyInstaller
freezes into PySPRESSO.exe so non-technical users can start the whole app
(backend + frontend) by double-clicking a single file, without Docker,
Python or a command line.
"""

import os
import threading
import time
import webbrowser

from pyspresso_app.config import APP_BASE_DIR
from pyspresso_app.workflows_api_methods import app

HOST = "127.0.0.1"
PORT = 5000


def _open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    # Match the working directory the operations rely on for relative
    # output/upload folders, whether running from source or frozen.
    os.chdir(APP_BASE_DIR)

    threading.Thread(target=_open_browser, daemon=True).start()

    print("=" * 60)
    print(" PySPRESSO")
    print(f" Opening automatically at http://{HOST}:{PORT}")
    print(" Keep this window open while using the app.")
    print(" Close this window (or press Ctrl+C) to stop PySPRESSO.")
    print("=" * 60)

    from waitress import serve

    serve(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
