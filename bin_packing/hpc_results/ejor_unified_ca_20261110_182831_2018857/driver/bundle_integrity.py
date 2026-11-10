"""Hash frozen text consistently across Windows checkout and Linux execution."""
import hashlib
from pathlib import Path

TEXT_SUFFIXES={'.py','.sh','.slurm','.md','.txt','.csv','.json'}


def normalized(data,path):
    return data.replace(b'\r\n',b'\n') if Path(path).suffix.lower() in TEXT_SUFFIXES else data


def verified_bytes(root,manifest,name):
    root=Path(root).resolve();path=(root/name).resolve();path.relative_to(root)
    data=path.read_bytes();expected=manifest['files_sha256'].get(name)
    canonical=normalized(data,path)
    candidates=[data,canonical]
    if path.suffix.lower() in TEXT_SUFFIXES:
        candidates.append(canonical.replace(b'\n',b'\r\n'))
    if not any(hashlib.sha256(candidate).hexdigest()==expected for candidate in candidates):
        raise ValueError('Frozen runtime/input changed: '+name)
    return canonical


def canonicalize_manifest(root,manifest):
    for name in list(manifest['files_sha256']):
        canonical=verified_bytes(root,manifest,name)
        manifest['files_sha256'][name]=hashlib.sha256(canonical).hexdigest()
    manifest['files_hash_policy']='LF-normalized text; exact binary bytes'
    return manifest
