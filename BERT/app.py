import os

from flask import Flask, jsonify
from flask_cors import CORS

# Eagerly import the pipeline so BERT + spaCy load at startup, not on first request.
import pipeline.analyzer  # noqa: F401


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# CORS: allow the Live Server origin to hit any /api/* route.
# If you serve the frontend from a different port, add it to the list.
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]}},
    supports_credentials=False,
)

# ── Config ───────────────────────────────────────────────────────────────────
# Limit uploads to 25 MB — enough for typical academic papers.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ── Blueprint registration ────────────────────────────────────────────────────
from api.assess  import assess_bp
from api.results import results_bp
from api.auth    import auth_bp
from api.user import user_bp
from api.history import history_bp

app.register_blueprint(assess_bp, url_prefix="/api")
app.register_blueprint(results_bp, url_prefix="/api")
app.register_blueprint(auth_bp, url_prefix="/api")
app.register_blueprint(user_bp, url_prefix="/api/user")
app.register_blueprint(history_bp, url_prefix="/api")


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


# ── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 25 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # debug=True auto-reloads on code changes but RELOADS THE BERT MODEL EACH TIME.
    # Turn off if you're iterating frequently and don't need auto-reload.
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=False)
