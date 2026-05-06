import re

from flask import Blueprint, g, jsonify, request
from firebase_admin import auth as fb_auth

from api.auth_utils import require_auth
from storage.firebase_store import (
    load_user_profile,
    update_user_profile,
    update_auth_email,
)


user_bp = Blueprint("user", __name__)

EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

ALLOWED_AVATAR_COLORS = {"amethyst", "rose", "teal"}
DEFAULT_AVATAR_COLOR  = "amethyst"

@user_bp.get("/profile")
@require_auth
def get_profile():
    profile = load_user_profile(g.uid)
    if profile is None:
        return jsonify({"error": "User profile not found"}), 404

    stored_color = profile.get("avatarColor")
    avatar_color = stored_color if stored_color in ALLOWED_AVATAR_COLORS else DEFAULT_AVATAR_COLOR

    return jsonify({
        "firstName":   profile.get("firstName", ""),
        "lastName":    profile.get("lastName", ""),
        "email":       profile.get("email", ""),
        "avatarColor": avatar_color,
    }), 200


@user_bp.put("/profile")
@require_auth
def update_profile():
    body = request.get_json(silent=True) or {}

    first_name = (body.get("firstName") or "").strip()
    last_name  = (body.get("lastName")  or "").strip()
    new_email  = (body.get("email")     or "").strip().lower()
    avatar_color = (body.get("avatarColor") or DEFAULT_AVATAR_COLOR).strip().lower()

    # validation
    if not first_name:
        return jsonify({"error": "First name is required"}), 400
    if not last_name:
        return jsonify({"error": "Last name is required"}), 400
    if not EMAIL_REGEX.match(new_email):
        return jsonify({"error": "Invalid email format"}), 400
    if avatar_color not in ALLOWED_AVATAR_COLORS:
        avatar_color = DEFAULT_AVATAR_COLOR

    # load current profile
    current = load_user_profile(g.uid)
    if current is None:
        return jsonify({"error": "User profile not found"}), 404

    current_email = (current.get("email") or "").lower()
    email_changed = new_email != current_email

    # update Auth email first
    if email_changed:
        try:
            update_auth_email(g.uid, new_email)
        except fb_auth.EmailAlreadyExistsError:
            return jsonify({"error": "That email is already in use by another account"}), 409
        except Exception as exc:
            return jsonify({"error": f"Failed to update email: {str(exc)}"}), 500

    # update Firestore profile
    try:
        update_user_profile(g.uid, {
            "firstName": first_name,
            "lastName":  last_name,
            "email":     new_email,
            "avatarColor": avatar_color,
        })
    except Exception as exc:
        # if Firestore fails
        if email_changed:
            try:
                update_auth_email(g.uid, current_email)
            except Exception:
                pass
        return jsonify({"error": f"Failed to update profile: {str(exc)}"}), 500

    return jsonify({
        "message": "Profile updated successfully",
        "emailChanged": email_changed,
        "profile": {
            "firstName": first_name,
            "lastName":  last_name,
            "email":     new_email,
            "avatarColor": avatar_color,
        }
    }), 200