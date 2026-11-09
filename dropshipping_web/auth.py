from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from typing import Optional
from database import veritabani_al
from models import Kullanici
from datetime import datetime, timedelta
import bcrypt

GIZLI_ANAHTAR = "cok-gizli-anahtar-bunu-degistir-ve-env-kullan"
ALGORITMA = "HS256"
TOKEN_SURESI_DAKIKA = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def sifre_hashle(sifre: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(sifre.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def sifre_dogrula(sifre: str, hash: str) -> bool:
    return bcrypt.checkpw(sifre.encode("utf-8"), hash.encode("utf-8"))

def erisim_tokeni_olustur(kullanici_adi: str, kullanici_id: int, rol: str):
    sona_erme = datetime.utcnow() + timedelta(minutes=TOKEN_SURESI_DAKIKA)
    payload = {
        "alt": kullanici_adi,
        "id": kullanici_id,
        "rol": rol,
        "exp": sona_erme
    }
    return jwt.encode(payload, GIZLI_ANAHTAR, algorithm=ALGORITMA)

def mevcut_kullanici(
    token: Optional[str] = Depends(oauth2_scheme),
    request: Request = None,
    db: Session = Depends(veritabani_al)
):
    if not token and request:
        token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Token bulunamadi")
    try:
        payload = jwt.decode(token, GIZLI_ANAHTAR, algorithms=[ALGORITMA])
        kullanici_adi: str = payload.get("alt")
        if kullanici_adi is None:
            raise HTTPException(status_code=401, detail="Gecersiz token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Gecersiz token")
    kullanici = db.query(Kullanici).filter(Kullanici.kullanici_adi == kullanici_adi).first()
    if kullanici is None:
        raise HTTPException(status_code=401, detail="Kullanici bulunamadi")
    return kullanici

def mevcut_kullanici_opsiyonel(
    token: Optional[str] = Depends(oauth2_scheme),
    request: Request = None,
    db: Session = Depends(veritabani_al)
):
    try:
        return mevcut_kullanici(token, request, db)
    except HTTPException:
        return None

def admin_mi(kullanici: Kullanici = Depends(mevcut_kullanici)):
    if kullanici.rol != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erisim")
    return kullanici