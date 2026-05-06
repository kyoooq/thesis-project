import os
import uuid
import time

from api.auth_utils import require_auth

from flask import Blueprint, jsonify, request, g

from pipeline.analyzer import analyze_paper
from storage.firebase_store import save_result, save_upload


assess_bp = Blueprint("assess", __name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def _get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


@assess_bp.post("/assess")
@require_auth
def assess():
    # ── Validate file presence ───────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Attach the file as 'file'."}), 400

    upload = request.files["file"]
    if not upload or upload.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = _get_extension(upload.filename)
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Use .pdf or .docx"
        }), 400

    # ── Persist the uploaded file ────────────────────────────────────────────
    result_id    = str(uuid.uuid4())
    saved_path   = save_upload(result_id, upload, ext)
    original_name = upload.filename

    # ── Run analysis ─────────────────────────────────────────────────────────
    try:
        result = analyze_paper(saved_path, threshold=0.7, save_csv=False)
    except Exception as exc:
        # Clean up the upload if analysis fails
        try:
            os.remove(saved_path)
        except OSError:
            pass
        return jsonify({
            "error": f"Analysis failed: {str(exc)}"
        }), 500

    # ── Attach metadata and persist ──────────────────────────────────────────
    result["resultId"] = result_id
    result["originalName"] = original_name
    result["ownerUid"] = g.uid
    result["createdAt"]    = int(time.time() * 1000)

    save_result(result_id, result)

    # ── Respond with the id (frontend uses it for the redirect) ──────────────
    return jsonify({"resultId": result_id}), 200
