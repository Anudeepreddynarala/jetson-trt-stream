import os
import json
import base64
import numpy as np
from cryptography.fernet import Fernet

KEY_FILE = "/home/user/arcface_project/secret.key"
DB_FILE  = "/home/user/arcface_project/embeddings.json"

def load_or_create_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    return key

fernet = Fernet(load_or_create_key())

def _load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def _save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def save_encrypted_embedding(name: str, emb: np.ndarray):
    db = _load_db()
    emb_bytes = emb.astype(np.float32).tobytes()
    token = fernet.encrypt(emb_bytes)
    db[name] = base64.b64encode(token).decode("utf-8")
    _save_db(db)

def load_all_embeddings():
    db = _load_db()
    result = {}
    for name, token_b64 in db.items():
        token = base64.b64decode(token_b64.encode("utf-8"))
        emb_bytes = fernet.decrypt(token)
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        result[name] = emb
    return result

def load_all_embeddings():
    """Load all enrolled embeddings"""
    if not os.path.exists(DB_FILE):
        return {}
    
    key = load_or_create_key()
    fernet = Fernet(key)
    
    with open(DB_FILE, 'r') as f:
        db = json.load(f)
    
    all_embeddings = {}
    for person_id, encrypted_data in db.items():
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted = fernet.decrypt(encrypted_bytes)
        embedding = np.frombuffer(decrypted, dtype=np.float32)
        all_embeddings[person_id] = embedding
    
    return all_embeddings
