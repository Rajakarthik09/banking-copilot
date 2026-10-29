"""Who is asking: the X-API-Key header is mapped to an rm_id on the server.

The request body cannot name an RM, so a client can't act as another RM by editing JSON.
Demo keys come from API_KEYS; in production this would be SSO / an OIDC JWT.
"""

import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def current_rm(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    for key, rm_id in get_settings().api_key_map().items():
        if x_api_key and hmac.compare_digest(x_api_key, key):
            return rm_id
    raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
