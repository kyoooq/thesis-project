from functools import wraps
from flask import request, jsonify, g

from storage.firebase_store import load_session


def require_auth(func):
    """
    Decorator that requires a valid session token.
    Puts the authenticated user's uid on `g.uid`.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authentication required."}), 401

        token = auth_header.removeprefix("Bearer ").strip()
        uid = load_session(token)

        if uid is None:
            return jsonify({"error": "Invalid or expired session."}), 401

        g.uid = uid
        return func(*args, **kwargs)

    return wrapper