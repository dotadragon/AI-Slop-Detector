from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core.config import settings


def ensure_rsa_keys() -> tuple[str, str]:
    private_path = Path(settings.jwt_private_key_path)
    public_path = Path(settings.jwt_public_key_path)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    if not private_path.exists() or not public_path.exists():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_bytes = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_bytes = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        private_path.write_bytes(private_bytes)
        try:
            private_path.chmod(0o600)
        except Exception:
            pass
        public_path.write_bytes(public_bytes)

    return private_path.read_text(), public_path.read_text()
