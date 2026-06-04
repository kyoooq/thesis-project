import re
import secrets
import urllib.request
import urllib.error
import json
from datetime import datetime

from flask import Blueprint, jsonify, request
from firebase_admin import auth as fb_auth

from storage.firebase_store import (
    create_auth_user,
    delete_auth_user,
    save_user_profile,
    save_session,
)


FIREBASE_WEB_API_KEY = "AIzaSyB1V59vTsrIJeqKNskbEDKynz4p9k1P1ms"
FIREBASE_SIGN_IN_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key="
    + FIREBASE_WEB_API_KEY
)


FIREBASE_SEND_OOB_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key="
    + FIREBASE_WEB_API_KEY
)


def _send_oob_email(email: str, request_type: str, password: str = None) -> tuple[bool, str]:
    print(f"\n>>> _send_oob_email CALLED: type={request_type} email={email}")

    # VERIFY_EMAIL requires an idToken; PASSWORD_RESET doesn't.
    if request_type == "VERIFY_EMAIL":
        if not password:
            print(">>> VERIFY_EMAIL requires password but none given")
            return False, "Password required to send verification email"

        # Get an idToken by signing in via REST
        signin_payload = json.dumps({
            "email": email,
            "password": password,
            "returnSecureToken": True,
        }).encode("utf-8")
        signin_req = urllib.request.Request(
            FIREBASE_SIGN_IN_URL,
            data=signin_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(signin_req, timeout=10) as resp:
                signin_body = json.loads(resp.read())
                id_token = signin_body.get("idToken")
                if not id_token:
                    print(">>> No idToken in signin response")
                    return False, "Could not obtain idToken"
        except urllib.error.HTTPError as e:
            print(f">>> Sign-in for idToken failed: HTTP {e.code}")
            return False, "Could not obtain idToken"

        payload = json.dumps({
            "requestType": "VERIFY_EMAIL",
            "idToken":     id_token,
        }).encode("utf-8")
    else:
        # PASSWORD_RESET — just needs email
        payload = json.dumps({
            "requestType": request_type,
            "email":       email,
        }).encode("utf-8")

    req = urllib.request.Request(
        FIREBASE_SEND_OOB_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            response_body = json.loads(resp.read())
            print(f">>> SUCCESS: {response_body}")
        return True, ""
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            error_msg = body.get("error", {}).get("message", "Unknown error")
            print(f">>> HTTPError {e.code}: {error_msg}")
            print(f">>> Full response: {body}")
            return False, error_msg
        except Exception:
            print(f">>> HTTPError {e.code}, unable to parse body")
            return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        print(f">>> URLError: {e}")
        return False, "Authentication service unavailable"
    except Exception as e:
        print(f">>> UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False, str(e)


auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
MIN_PASSWORD_LEN = 8


@auth_bp.post("/auth/register")
def register():

    data = request.get_json(silent=True) or {}

    first_name = (data.get("firstName") or "").strip()
    last_name  = (data.get("lastName")  or "").strip()
    email      = (data.get("email")     or "").strip().lower()
    password   = data.get("password")   or ""

    # validate
    if not first_name or not last_name:
        return jsonify({"error": "First name and last name are required."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please provide a valid email address."}), 400
    if len(password) < MIN_PASSWORD_LEN:
        return jsonify({
            "error": f"Password must be at least {MIN_PASSWORD_LEN} characters."
        }), 400

    # create in firebase auth
    try:
        uid = create_auth_user(email, password)
    except fb_auth.EmailAlreadyExistsError:
        return jsonify({"error": "An account with this email already exists."}), 409
    except Exception as exc:
        return jsonify({"error": f"Registration failed: {str(exc)}"}), 500

    # save profile to firestore 
    try:
        save_user_profile(uid, {
            "email":     email,
            "firstName": first_name,
            "lastName":  last_name,
            "createdAt": datetime.utcnow().isoformat(),
        })
    except Exception as exc:
        # Rollback if profile save fails, remove the Auth user too
        delete_auth_user(uid)
        return jsonify({"error": f"Registration failed: {str(exc)}"}), 500

    _send_oob_email(email, "VERIFY_EMAIL", password=password)

    return jsonify({
        "success": True,
        "uid": uid,
        "message": "Account created. Please check your email to verify your address."
    }), 201



@auth_bp.post("/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    # verify with firebase Auth REST API
    payload = json.dumps({
        "email":             email,
        "password":          password,
        "returnSecureToken": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        FIREBASE_SIGN_IN_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return jsonify({"error": "Invalid email or password."}), 401
    except urllib.error.URLError:
        return jsonify({"error": "Authentication service unavailable."}), 503

    uid = body.get("localId")
    if not uid:
        return jsonify({"error": "Login failed."}), 500

    # Reject login if email is not verified
    try:
        user = fb_auth.get_user(uid)
        if not user.email_verified:
            # Resend verification email since we have credentials in hand
            _send_oob_email(email, "VERIFY_EMAIL", password=password)
            return jsonify({
                "error": "Your email is not verified. We've sent a new verification link to your inbox.",
                "code": "EMAIL_NOT_VERIFIED",
                "email": email,
            }), 403
    except Exception:
        return jsonify({"error": "Login failed."}), 500

    # issue session token
    token = secrets.token_urlsafe(32)
    save_session(token, uid)

    # issue session token
    token = secrets.token_urlsafe(32)
    save_session(token, uid)

    return jsonify({"token": token, "uid": uid}), 200



@auth_bp.post("/auth/logout")
def logout():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "No token provided."}), 400

    token = auth_header.removeprefix("Bearer ").strip()

    from storage.firebase_store import delete_session
    delete_session(token)

    return jsonify({"success": True}), 200


@auth_bp.post("/auth/forgot-password")
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    _send_oob_email(email, "PASSWORD_RESET")

    return jsonify({
        "success": True,
        "message": "If an account exists for this email, a password reset link has been sent."
    }), 200


@auth_bp.post("/auth/resend-verification")
def resend_verification():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    _send_oob_email(email, "VERIFY_EMAIL")

    return jsonify({
        "success": True,
        "message": "If an account exists for this email, a verification link has been sent."
    }), 200