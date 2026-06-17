import os

from flask import Flask, jsonify
from flask_cors import CORS

import pipeline.analyzer


# Flask
app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]}},
    supports_credentials=False,
)

# Config (file size)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


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


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


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
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=False)
