from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# Deliberately not pydantic's EmailStr — that needs the `email-validator`
# extra, a real dependency for a check this project only needs to be
# "obviously an email-shaped string," not RFC 5322-exact.
_EMAIL_MAX_LENGTH = 320


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=_EMAIL_MAX_LENGTH)
    password: str = Field(min_length=8, max_length=200)

    @field_validator("email")
    @classmethod
    def looks_like_an_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("email must look like an email address")
        return v


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
