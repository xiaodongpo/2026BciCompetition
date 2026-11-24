"""The owner-approved model exception must not admit unrelated private artifacts."""
import hashlib,json
import pytest
from scripts import check_release

@pytest.fixture
def publication(tmp_path,monkeypatch):
    monkeypatch.setattr(check_release,'ROOT',tmp_path)
    (tmp_path/'LICENSE').write_text('MIT License',encoding='utf8')
    model=tmp_path/'checkpoints/h1/fixture.pt';model.parent.mkdir(parents=True)
    model.write_bytes(b'\x00synthetic model fixture')
    manifest=[dict(route='h1',filename='fixture.pt',published_path='checkpoints/h1/fixture.pt',owner_approved=True,bytes=model.stat().st_size,sha256=hashlib.sha256(model.read_bytes()).hexdigest())]
    (tmp_path/'checkpoints/manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    return tmp_path,model

def test_approved_model_digest(publication):
    report=check_release.scan()
    assert report['status']=='PASS' and report['approved_model_files']==1

def test_altered_model_is_rejected(publication):
    _,model=publication;model.write_bytes(b'changed')
    assert any(r['kind']=='approved_model_hash_mismatch' for r in check_release.scan()['findings'])

def test_unapproved_raw_data_is_rejected(publication):
    root,_=publication;(root/'unapproved.bdf').write_bytes(b'\x00unapproved raw data fixture')
    assert any(r['kind']=='private_artifact' for r in check_release.scan()['findings'])
