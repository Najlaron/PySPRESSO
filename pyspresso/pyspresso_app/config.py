import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from pyspresso_app.bootstrap import initialize

# When packaged as a standalone executable (PyInstaller), user data (database,
# uploads, outputs) must live next to the .exe instead of inside the
# temporary folder the app is unpacked into. In normal (non-frozen) runs this
# resolves to the "pyspresso" project folder, matching the previous behaviour.
if getattr(sys, "frozen", False):
    APP_BASE_DIR = Path(sys.executable).resolve().parent
    # Packaged exe is self-contained: uploads live right next to it too.
    UPLOADS_BASE_DIR = APP_BASE_DIR
else:
    APP_BASE_DIR = Path(__file__).resolve().parent.parent
    # Matches the original layout: uploads/ sits at the repo root, one level
    # above the pyspresso/ folder (same in local dev and in Docker, where
    # WORKDIR /app is mounted from ./pyspresso).
    UPLOADS_BASE_DIR = APP_BASE_DIR.parent

INSTANCE_DIR = APP_BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

# Pre-built frontend assets (produced by `npm run build`) served by Flask so
# the whole app can run as a single process/executable without Node.js.
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend_dist"

app = Flask(
    __name__,
    instance_path=str(INSTANCE_DIR),
    static_folder=str(FRONTEND_DIST_DIR),
    static_url_path="",
)
CORS(app)

db_path = INSTANCE_DIR / "workflows.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path.as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# Serves the React SPA's index.html for any route Flask didn't match as an
# API endpoint or a real static file (e.g. client-side routes like
# /workflow/<id> or a browser refresh on a deep link).
@app.errorhandler(404)
def _serve_frontend_or_404(error):
    index_file = FRONTEND_DIST_DIR / "index.html"
    if request.method == "GET" and index_file.is_file():
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")
    return jsonify({"message": "Not found"}), 404


initialize()
