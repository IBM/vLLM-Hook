"""Small-file transfers and validation for the pinned Triton CPU wheel."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile
from email.parser import Parser

CHUNK_SIZE = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_wheel(path, revision):
    path = Path(path)
    if not path.name.endswith('.whl'):
        raise ValueError('Invalid wheel filename')
    with zipfile.ZipFile(path) as archive:
        metadata_names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        wheel_names = [n for n in archive.namelist() if n.endswith('.dist-info/WHEEL')]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError('Wheel must contain one distribution')
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode())
        tags = Parser().parsestr(archive.read(wheel_names[0]).decode()).get_all('Tag', [])
    if metadata['Name'].lower() != 'triton' or ('cpu.git' + revision[:8]) not in metadata['Version']:
        raise ValueError('Wheel does not match the pinned Triton CPU revision')
    if not any(t.startswith('cp312-cp312-') and 'linux' in t and t.endswith('x86_64') for t in tags):
        raise ValueError('Expected a Linux x86_64 CPython 3.12 wheel')
    return metadata['Version']


def export_chunks(wheel, revision, destination):
    wheel, destination = Path(wheel), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    version = validate_wheel(wheel, revision)
    chunks = []
    with wheel.open('rb') as stream:
        for number, block in enumerate(iter(lambda: stream.read(CHUNK_SIZE), b'')):
            part = destination / f'triton-{number:04d}.part'
            part.write_bytes(block)
            chunks.append({'path': str(part), 'sha256': hashlib.sha256(block).hexdigest()})
    manifest = {'revision': revision, 'filename': wheel.name, 'sha256': sha256(wheel),
                'version': version, 'python': 'cp312', 'platform': 'linux-x86_64',
                'libc': list(platform.libc_ver()), 'chunks': chunks}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_chunks(manifest, parts, destination):
    name = manifest['filename']
    if Path(name).name != name or not name.endswith('.whl'):
        raise ValueError('Unsafe wheel filename')
    if len(parts) != len(manifest['chunks']):
        raise ValueError('Missing wheel chunks')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / (name + '.partial')
    try:
        with temporary.open('wb') as output:
            for path, expected in zip(parts, manifest['chunks']):
                if sha256(path) != expected['sha256']:
                    raise ValueError('Wheel chunk checksum mismatch')
                with Path(path).open('rb') as source:
                    while block := source.read(CHUNK_SIZE):
                        output.write(block)
        if sha256(temporary) != manifest['sha256']:
            raise ValueError('Wheel checksum mismatch')
        result = destination / name
        temporary.replace(result)
        validate_wheel(result, manifest['revision'])
        return result
    finally:
        temporary.unlink(missing_ok=True)
