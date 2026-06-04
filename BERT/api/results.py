import re

from api.auth_utils import require_auth

from flask import Blueprint, jsonify, g

from storage.firebase_store import load_result


results_bp = Blueprint("results", __name__)

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

@results_bp.get("/results/<result_id>")
@require_auth
def get_result(result_id: str):
    if not ID_PATTERN.fullmatch(result_id):
        return jsonify({"error": "Invalid result id."}), 400

    result = load_result(result_id)
    if result is None:
        return jsonify({"error": "Result not found."}), 404
    
    if result.get("ownerUid") != g.uid:
        return jsonify({"error": "Result not found."}), 404

    return jsonify(result), 200
