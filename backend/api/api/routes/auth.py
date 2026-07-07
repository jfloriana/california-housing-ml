from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr

from ..utils.supabase_client import get_client, get_service_client

router = APIRouter()


class SignUpRequest(BaseModel):
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class SignInResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: Optional[str] = None


class PreferencesResponse(BaseModel):
    language: str


class PreferencesUpdateRequest(BaseModel):
    language: str


class VerifyTokenResponse(BaseModel):
    valid: bool
    user: Optional[dict] = None


def _get_user_id_from_token(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        client = get_client()
        user = client.auth.get_user(token)
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@router.post("/api/auth/signup", response_model=UserResponse)
async def signup(request: SignUpRequest):
    try:
        client = get_client()
        result = client.auth.sign_up({"email": request.email, "password": request.password})
        user = result.user
        if user is None:
            raise HTTPException(status_code=400, detail="Signup failed")

        try:
            svc = get_service_client()
            svc.table("user_preferences").insert({
                "user_id": user.id,
                "language": "es",
            }).execute()
        except Exception:
            pass

        return UserResponse(id=user.id, email=user.email)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup error: {str(e)}")


@router.post("/api/auth/signin", response_model=SignInResponse)
async def signin(request: SignInRequest):
    try:
        client = get_client()
        result = client.auth.sign_in_with_password({"email": request.email, "password": request.password})
        session = result.session
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = result.user
        return SignInResponse(
            access_token=session.access_token,
            user={"id": user.id, "email": user.email},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Signin error: {str(e)}")


@router.post("/api/auth/signout")
async def signout(authorization: str = Header(None)):
    try:
        client = get_client()
        client.auth.sign_out()
        return {"message": "Signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signout error: {str(e)}")


@router.get("/api/auth/user", response_model=UserResponse)
async def get_user(user_id: str = Depends(_get_user_id_from_token)):
    try:
        svc = get_service_client()
        result = svc.auth.admin.get_user_by_id(user_id)
        user = result.user
        return UserResponse(id=user.id, email=user.email)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")


@router.get("/api/auth/preferences", response_model=PreferencesResponse)
async def get_preferences(user_id: str = Depends(_get_user_id_from_token)):
    try:
        svc = get_service_client()
        result = svc.table("user_preferences").select("language").eq("user_id", user_id).execute()
        if result.data and len(result.data) > 0:
            return PreferencesResponse(language=result.data[0].get("language", "es"))
        return PreferencesResponse(language="es")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.put("/api/auth/preferences", response_model=PreferencesResponse)
async def update_preferences(
    request: PreferencesUpdateRequest,
    user_id: str = Depends(_get_user_id_from_token),
):
    if request.language not in ("es", "en"):
        raise HTTPException(status_code=400, detail="Language must be 'es' or 'en'")

    try:
        svc = get_service_client()
        existing = svc.table("user_preferences").select("id").eq("user_id", user_id).execute()

        if existing.data and len(existing.data) > 0:
            svc.table("user_preferences").update({"language": request.language}).eq("user_id", user_id).execute()
        else:
            svc.table("user_preferences").insert({
                "user_id": user_id,
                "language": request.language,
            }).execute()

        return PreferencesResponse(language=request.language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.post("/api/auth/verify", response_model=VerifyTokenResponse)
async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return VerifyTokenResponse(valid=False)
    token = authorization.split(" ", 1)[1]
    try:
        client = get_client()
        user = client.auth.get_user(token)
        if user and user.user:
            return VerifyTokenResponse(
                valid=True,
                user={"id": user.user.id, "email": user.user.email},
            )
        return VerifyTokenResponse(valid=False)
    except Exception:
        return VerifyTokenResponse(valid=False)
