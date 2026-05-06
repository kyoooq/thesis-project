import re

from flask import Blueprint, jsonify, g

from storage.firebase_store import list_results, load_result, delete_result
from api.auth_utils import require_auth


history_bp = Blueprint("history", __name__)

# Same id pattern used in results.py — reject obviously malformed ids early
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@history_bp.get("/history")
@require_auth
def get_history():
    """
    Return all assessments owned by the authenticated user, newest first.
    Response shape matches what history.html expects:
        [
          {
            "id":           "<uuid>",
            "filename":     "paper.pdf",
            "overallScore": 74,
            "overallLabel": "74% RESPONSIVE",
            "createdAt":    1713700000000,
            "rows":         [ ...same shape as results page... ]
          },
          ...
        ]
    """
    results = list_results(owner_uid=g.uid)

    history = []
    for r in results:
        history.append({
            "id":           r.get("resultId"),
            "filename":     r.get("originalName"),
            "overallScore": r.get("overallScore"),
            "overallLabel": r.get("overallLabel"),
            "createdAt":    r.get("createdAt"),
            "rows":         r.get("rows", []),
        })

    return jsonify(history), 200


@history_bp.delete("/history/<result_id>")
@require_auth
def delete_history_item(result_id: str):
    """
    Delete one of the caller's assessments. Returns 404 (not 403) if the result
    exists but belongs to someone else — this prevents leaking the existence of
    other users' result IDs.
    """
    if not ID_PATTERN.fullmatch(result_id):
        return jsonify({"error": "Invalid result id."}), 400

    result = load_result(result_id)
    if result is None:
        return jsonify({"error": "Result not found."}), 404

    # Ownership check — identical pattern to api/results.py
    if result.get("ownerUid") != g.uid:
        return jsonify({"error": "Result not found."}), 404

    delete_result(result_id)
    return jsonify({"success": True}), 200