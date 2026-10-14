"""The entry bundle (lib/entry_bundle.py): what is written, and that a bundle is unpacked only when it checks.

A bundle is made under a member's account on a cluster and unpacked on a server, so most of this file hands
:func:`lib.entry_bundle.open_bundle` bundles that were changed after they were written.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from lib import archive, entry_bundle, entry_run
from lib.entry_bundle import BundleError

import synthetic_run
from synthetic_glycan import GLYCAM_NAME

LEVEL_FILES = ["dist.svg", "GLYCAM/alpha.pdb", "GLYCAM/beta.pdb", "PDB/alpha.pdb", "PDB/beta.pdb"]
MEMBERS = [
    "bundle.json",
    "archive/trajectory.xtc", "archive/topology.pdb", "archive/structure.mol2", "archive/manifest.json",
    "archive/info.json", "archive/name.json",
    *sorted([
        "static/landscape/landscape_metadata.json", "static/landscape/pca_model.joblib",
        "static/landscape/preprocessing.npz", "static/output/info.json", "static/output/torparts.npz",
        "static/output/pca.csv", "static/output/torsion_all.csv", "static/output/torsion_glycosidic.csv",
        "static/output/torsion_level_1_reps.csv", "static/output/torsion_level_2_reps.csv",
        "static/output/replicas.json",
        *[f"static/output/level_{level}/{name}" for level in (1, 2) for name in LEVEL_FILES],
    ]),
    "result.json", "run.log",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return synthetic_run.get(tmp_path_factory)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Bundle:
    """The members of a bundle in memory, to change them and write them back."""

    def __init__(self, path):
        self.members = {}
        with tarfile.open(path) as tar:
            for member in tar:
                self.members[member.name] = tar.extractfile(member).read()
        self.description = json.loads(self.members.pop("bundle.json"))

    def json(self, name):
        return json.loads(self.members[name])

    def set_json(self, name, value):
        self.members[name] = (json.dumps(value, indent=2) + "\n").encode()

    def refresh(self, order=True):
        """``files`` of bundle.json from the members as they are now."""
        files = {
            name: {"size": len(data), "sha256": sha256(data)}
            for name, data in self.members.items() if isinstance(data, bytes)
        }
        names = entry_bundle.member_order(list(files)) if order else list(files)
        names += [name for name in files if name not in names]
        self.description["files"] = {name: files[name] for name in names}
        self.members = {**{name: self.members[name] for name in names if name in self.members},
                        **{name: data for name, data in self.members.items() if name not in names}}
        return self

    def refresh_result(self):
        """Make result.json describe the manifest and the static files as they are now (a consistent forgery)."""
        result = self.json("result.json")
        manifest = self.json("archive/manifest.json")
        result["input"]["files"] = manifest["files"]
        result["input"]["manifest_sha256"] = sha256(self.members["archive/manifest.json"])
        result["produced"]["static"]["files"] = [
            {"path": name[len("static/"):], "size": len(data), "sha256": sha256(data)}
            for name, data in sorted(self.members.items()) if name.startswith("static/")
        ]
        self.set_json("result.json", result)
        return self

    def write(self, path, description=None, first=None, fmt=tarfile.USTAR_FORMAT):
        """Write the tar: bundle.json (or ``description`` bytes), then the members; a TarInfo is added as it is."""
        payload = description if description is not None else (json.dumps(self.description, indent=2) + "\n").encode()
        entries = [("bundle.json", payload), *self.members.items()]
        if first is not None:
            entries.insert(0, first)
        with tarfile.open(path, "w", format=fmt) as tar:
            for name, data in entries:
                if isinstance(data, tarfile.TarInfo):
                    tar.addfile(data)
                else:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        return Path(path)


def refused(tmp_path, tar_path, message, **options):
    """open_bundle and check_bundle refuse with ``message`` and nothing is left behind."""
    target = tmp_path / "opened"
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(BundleError, match=message):
        entry_bundle.open_bundle(tar_path, target, **options)
    assert not target.exists() and sorted(path.name for path in tmp_path.iterdir()) == before
    with pytest.raises(BundleError, match=message):
        entry_bundle.check_bundle(tar_path, **options)


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------
def test_names_of_the_archive_and_the_result_are_those_of_their_modules():
    assert entry_bundle.ARCHIVE_SCHEMA == archive.ARCHIVE_SCHEMA
    assert entry_bundle.ARCHIVE_SCHEMA_VERSION == archive.SCHEMA_VERSION
    assert entry_bundle.RESULT_SCHEMA == entry_run.RESULT_SCHEMA
    assert (entry_bundle.RESULT_FILE, entry_bundle.LOG_FILE) == (entry_run.RESULT_FILE, entry_run.LOG_FILE)
    assert entry_bundle.MANIFEST_FILE == archive.MANIFEST_FILE
    assert set(entry_bundle.ARCHIVE_FILES) == {
        archive.TRAJECTORY_FILE, archive.TOPOLOGY_FILE, archive.MOL2_FILE, archive.MANIFEST_FILE, archive.INFO_FILE,
        archive.NAME_FILE}


def test_bundle_is_a_plain_tar_with_the_members_of_the_contract_in_order(run):
    with tarfile.open(run["bundle"]) as tar:
        members = tar.getmembers()
        assert [member.name for member in members] == MEMBERS
        assert all(member.isreg() and member.mode == 0o644 and (member.uid, member.gid) == (0, 0)
                   and (member.uname, member.gname) == ("", "") for member in members)
        assert tar.pax_headers == {} and all(member.pax_headers == {} for member in members)
        content = {member.name: tar.extractfile(member).read() for member in members}
    raw = run["bundle"].read_bytes()
    assert raw[:11] == b"bundle.json" and raw[257:265] == b"ustar\x0000"       # a plain ustar header, not compressed
    assert len(raw) % 10240 == 0

    bundle = json.loads(content["bundle.json"])
    assert list(bundle) == ["schema", "schema_version", "glycam_name", "created", "pipeline", "levels", "files",
                            "source", "extends"]
    assert (bundle["schema"], bundle["schema_version"]) == ("glycoshape-entry-bundle", 1)
    assert bundle["glycam_name"] == GLYCAM_NAME and bundle["levels"] == [1, 2] and bundle["extends"] is None
    assert bundle["created"].endswith("Z") and len(bundle["created"]) == 20
    assert list(bundle["files"]) == MEMBERS[1:]
    for name, record in bundle["files"].items():
        assert record == {"size": len(content[name]), "sha256": sha256(content[name])}

    # the entry and the result are the files of the work folder, unchanged
    entry = run["work"] / "archive" / GLYCAM_NAME
    for name in ("trajectory.xtc", "topology.pdb", "structure.mol2", "manifest.json", "info.json", "name.json"):
        assert content[f"archive/{name}"] == (entry / name).read_bytes()
    assert content["result.json"] == (run["work"] / "result" / "result.json").read_bytes()
    assert content["run.log"] == (run["work"] / "result" / "run.log").read_bytes()
    static = run["work"] / "result" / "static"
    for name in MEMBERS:
        if name.startswith("static/"):
            assert content[name] == (static / name[len("static/"):]).read_bytes()

    result = json.loads(content["result.json"])
    manifest = json.loads(content["archive/manifest.json"])
    pipeline = bundle["pipeline"]
    assert list(pipeline) == ["commit", "uncommitted_changes", "version", "version_of", "converter_version", "python",
                              "packages"]
    assert pipeline["version"] == result["pipeline"]["code_sha256"] and len(pipeline["version"]) == 64
    assert pipeline["commit"] == result["pipeline"]["commit"]
    assert pipeline["converter_version"] == manifest["converter"]["version"] == archive.CONVERTER_VERSION
    assert run["made"]["bundle"] == {
        "path": str(run["bundle"]), "size": run["bundle"].stat().st_size, "sha256": sha256(run["bundle"].read_bytes())}


def test_bundle_unpacks_into_a_new_folder_with_that_layout(run, tmp_path):
    target = tmp_path / "incoming" / "bundle-1"
    assert entry_bundle.open_bundle(run["bundle"], target, expected_glycam_name=GLYCAM_NAME) == target
    files = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
    assert files == sorted(MEMBERS)
    with tarfile.open(run["bundle"]) as tar:
        for member in tar:
            unpacked = target / member.name
            assert unpacked.read_bytes() == tar.extractfile(member).read()
            assert stat.S_IMODE(unpacked.stat().st_mode) & 0o133 == 0 and not unpacked.is_symlink()
    assert sorted(path.name for path in target.parent.iterdir()) == ["bundle-1"]       # no partial folder is left
    bundle = entry_bundle.load_bundle_dir(target)
    assert bundle == json.loads((target / "bundle.json").read_text()) == entry_bundle.check_bundle(run["bundle"])

    # the archive entry of the bundle is an entry; its folder is not named by its manifest
    report = archive.verify_entry(target / "archive", check_folder=False)
    assert report["ok"] and report["frames"] == 480 and report["segments_checked"] == 3
    assert archive.ArchiveEntry(target / "archive").glycam_name == GLYCAM_NAME

    # an empty folder is accepted, anything else is not written into
    empty = tmp_path / "empty"
    empty.mkdir()
    assert entry_bundle.open_bundle(run["bundle"], empty) == empty and (empty / "bundle.json").is_file()
    with pytest.raises(BundleError, match="exists; a bundle is unpacked into a new folder"):
        entry_bundle.open_bundle(run["bundle"], empty)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(BundleError, match="exists; a bundle is unpacked into a new folder"):
        entry_bundle.open_bundle(run["bundle"], link)
    assert not (tmp_path / "elsewhere").exists()
    refused(tmp_path / "incoming", run["bundle"], "the bundle is 'DManpa1-2DManpa1-OH', 'DGlcpa1-OH' was expected",
            expected_glycam_name="DGlcpa1-OH")
    with pytest.raises(BundleError, match="cannot open"):
        entry_bundle.open_bundle(tmp_path / "missing.tar", tmp_path / "x")

    # the unpacked folder is checked again by whoever is handed it
    (target / "static" / "output" / "pca.csv").write_bytes(b"frame,PC1\n")
    with pytest.raises(BundleError, match="static/output/pca.csv is not the file bundle.json lists"):
        entry_bundle.load_bundle_dir(target)
    (empty / "static" / "added.txt").write_text("x")
    with pytest.raises(BundleError, match="does not hold exactly the files bundle.json lists"):
        entry_bundle.load_bundle_dir(empty)


def test_writing_is_reproducible_and_refuses_what_is_not_a_finished_result(run, tmp_path):
    entry = run["work"] / "archive" / GLYCAM_NAME
    source = json.loads(json.dumps(entry_bundle.check_bundle(run["bundle"])["source"]))
    created = "2026-10-11T20:00:00Z"
    first = entry_bundle.write_bundle(tmp_path / "a.tar", entry, run["work"] / "result", source, created=created)
    second = entry_bundle.write_bundle(tmp_path / "b.tar", entry, run["work"] / "result", source, created=created)
    assert (tmp_path / "a.tar").read_bytes() == (tmp_path / "b.tar").read_bytes()
    assert first["sha256"] == second["sha256"] and first["bundle"]["created"] == created
    with tarfile.open(tmp_path / "a.tar") as tar:
        assert {member.mtime for member in tar} == {1791748800}

    result_dir = tmp_path / "result"
    shutil.copytree(run["work"] / "result", result_dir, ignore=shutil.ignore_patterns("process"))

    def write(message):
        with pytest.raises(BundleError, match=message):
            entry_bundle.write_bundle(tmp_path / "out.tar", entry, result_dir, source)
        assert sorted(path.name for path in tmp_path.iterdir()) == ["a.tar", "b.tar", "result"]

    pca = result_dir / "static" / "output" / "pca.csv"
    original = pca.read_bytes()
    pca.write_bytes(original[:-1] + b"9")
    write("pca.csv is not the file that was recorded")
    pca.write_bytes(original + b"1\n")
    write(r"pca.csv has \d+ bytes, \d+ were recorded")
    pca.write_bytes(original)
    (result_dir / "static" / "notes.txt").write_text("added by hand")
    write("are not those result.json records")
    (result_dir / "static" / "notes.txt").unlink()
    pca.unlink()
    pca.symlink_to("/etc/hostname")
    write("is not a regular file")
    pca.unlink()
    pca.write_bytes(original)
    result = json.loads((result_dir / "result.json").read_text())
    (result_dir / "result.json").write_text(json.dumps(dict(result, status="failed")))
    write("has status 'failed': there is nothing to bundle")
    (result_dir / "result.json").write_text(json.dumps(dict(result, glycam_name="DGlcpa1-OH")))
    write("result.json is the result of 'DGlcpa1-OH'")
    (result_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(BundleError, match="source.simulation_length_ns is -1"):
        entry_bundle.write_bundle(tmp_path / "out.tar", entry, result_dir, dict(source, simulation_length_ns=-1))
    with pytest.raises(BundleError, match="created must look like"):
        entry_bundle.write_bundle(tmp_path / "out.tar", entry, result_dir, source, created="yesterday")
    assert entry_bundle.write_bundle(tmp_path / "out.tar", entry, result_dir, source)["bundle"]["glycam_name"] == GLYCAM_NAME


# ---------------------------------------------------------------------------
# Bundles that were changed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("member", ["archive/trajectory.xtc", "archive/manifest.json", "static/output/torsion_all.csv",
                                    "static/output/level_1/PDB/alpha.pdb", "result.json", "run.log"])
def test_changed_byte_is_noticed_in_every_member(run, tmp_path, member):
    bundle = Bundle(run["bundle"])
    data = bytearray(bundle.members[member])
    data[len(data) // 2] ^= 0x01
    bundle.members[member] = bytes(data)
    refused(tmp_path, bundle.write(tmp_path / "changed.tar"), f"member {member} does not have the SHA-256 that bundle.json lists")


def test_missing_member(run, tmp_path):
    bundle = Bundle(run["bundle"])
    del bundle.members["static/output/pca.csv"]
    refused(tmp_path, bundle.write(tmp_path / "a.tar"),
            r"member static/output/replicas.json comes where static/output/pca.csv is expected \(missing or out of order\)")
    bundle = Bundle(run["bundle"])
    del bundle.members["run.log"]
    refused(tmp_path, bundle.write(tmp_path / "b.tar"), "member run.log is missing: the archive ends before it")
    # ... left out of bundle.json as well
    for name, message in (
        ("static/output/pca.csv", "bundle.json does not list static/output/pca.csv"),
        ("static/output/level_2/GLYCAM/beta.pdb", "bundle.json does not list static/output/level_2/GLYCAM/beta.pdb"),
        ("archive/topology.pdb", "bundle.json does not list archive/topology.pdb"),
        ("result.json", "bundle.json does not list result.json"),
        ("static/output/torsion_all.csv", r"the static files of the bundle are not those result.json records \(1 differ, "
                                          "for example static/output/torsion_all.csv"),
        ("archive/info.json", "archive/manifest.json lists info.json, name.json, structure.mol2, topology.pdb, "
                              "trajectory.xtc; the bundle has name.json, structure.mol2"),
    ):
        bundle = Bundle(run["bundle"])
        del bundle.members[name]
        refused(tmp_path, bundle.refresh().write(tmp_path / "c.tar"), message)


def test_extra_member(run, tmp_path):
    bundle = Bundle(run["bundle"])
    bundle.members["static/extra.txt"] = b"not listed"
    refused(tmp_path, bundle.write(tmp_path / "a.tar"),
            r"member 'static/extra.txt' is not listed in bundle.json \(the archive should end here\)")
    bundle = Bundle(run["bundle"])
    bundle.members = {"static/extra.txt": b"not listed", **bundle.members}
    refused(tmp_path, bundle.write(tmp_path / "b.tar"),
            r"member 'static/extra.txt' is not listed in bundle.json \(archive/trajectory.xtc is expected here\)")
    # the same member twice
    bundle = Bundle(run["bundle"])
    with tarfile.open(tmp_path / "c.tar", "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in [("bundle.json", (json.dumps(bundle.description)).encode()), *bundle.members.items(),
                           ("run.log", b"another log")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    refused(tmp_path, tmp_path / "c.tar", r"member 'run.log' is not listed in bundle.json \(the archive should end here\)")
    # listed in bundle.json, with its hash: not what the result records
    bundle = Bundle(run["bundle"])
    bundle.members["static/extra.txt"] = b"listed"
    refused(tmp_path, bundle.refresh().write(tmp_path / "d.tar"),
            r"the static files of the bundle are not those result.json records \(1 differ, for example static/extra.txt")
    for name in ("archive/extra.bin", "archive/sub/trajectory.xtc", "notes.txt", "process/data/coords.zarr", "bundle.json"):
        bundle = Bundle(run["bundle"])
        bundle.members[name] = b"x"
        refused(tmp_path, bundle.refresh().write(tmp_path / "e.tar"), "is not a file of a bundle")
    # a second bundle.json that is not listed
    bundle = Bundle(run["bundle"])
    bundle.members["bundle.json"] = b"{}"
    refused(tmp_path, bundle.write(tmp_path / "f.tar"), "member 'bundle.json' is not listed in bundle.json")


@pytest.mark.parametrize("path,message", [
    ("../evil.txt", "has the part '..'"),
    ("static/../../evil.txt", "has the part '..'"),
    ("static/output/../../../etc/cron.d/evil", "has the part '..'"),
    ("/etc/passwd", "is absolute"),
    ("static//double.txt", "has the part ''"),
    ("static/./dot.txt", "has the part '.'"),
    ("static/.hidden", "'.hidden' is not a plain file name"),
    ("static\\evil.txt", "is not a plain file name"),
    ("static/with space.txt", "is not a plain file name"),
    ("static/new\nline", "is not a plain file name"),
    ("static/~user", "is not a plain file name"),
    ("static/" + "a/" * 8 + "deep.txt", "more than 8 folders deep"),
    ("static/" + "x" * 200, "is longer than 200 characters"),
    ("", "is not a non-empty string"),
])
def test_paths_that_leave_the_folder_or_are_not_plain_are_refused(run, tmp_path, path, message):
    bundle = Bundle(run["bundle"])
    bundle.description["files"][path] = {"size": 4, "sha256": sha256(b"evil")}
    payload = (json.dumps(bundle.description) + "\n").encode()
    # the member itself is written under a harmless name: the listing alone is refused
    refused(tmp_path, bundle.write(tmp_path / "listed.tar", description=payload), message)
    assert not (tmp_path.parent / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()


def test_member_whose_name_leaves_the_folder_is_not_written(run, tmp_path):
    """Not listed in bundle.json: the name in the tar header alone must not be followed."""
    for name in ("../evil.txt", "/tmp/gap-evil.txt", "static/../../evil.txt"):
        bundle = Bundle(run["bundle"])
        members = dict(bundle.members)
        bundle.members = {name: b"evil"}
        bundle.members.update(members)
        refused(tmp_path, bundle.write(tmp_path / "a.tar"), "is not listed in bundle.json")
        bundle = Bundle(run["bundle"])
        bundle.members[name] = b"evil"
        refused(tmp_path, bundle.write(tmp_path / "b.tar"), "is not listed in bundle.json")
    assert not (tmp_path.parent / "evil.txt").exists() and not Path("/tmp/gap-evil.txt").exists()


def special(name, kind, linkname="", size=0):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    info.size = size
    return info


@pytest.mark.parametrize("kind,linkname,label", [
    (tarfile.SYMTYPE, "/etc/passwd", "symbolic link"),
    (tarfile.SYMTYPE, "../../outside", "symbolic link"),
    (tarfile.LNKTYPE, "archive/trajectory.xtc", "hard link"),
    (tarfile.DIRTYPE, "", "directory"),
    (tarfile.CHRTYPE, "", "character device"),
    (tarfile.BLKTYPE, "", "block device"),
    (tarfile.FIFOTYPE, "", "FIFO"),
    (tarfile.CONTTYPE, "", "contiguous file"),
    (tarfile.GNUTYPE_SPARSE, "", "GNU sparse file"),
])
def test_links_devices_and_directories_are_refused(run, tmp_path, kind, linkname, label):
    # in place of a listed member, and as an extra one before the members
    bundle = Bundle(run["bundle"])
    bundle.members["static/output/pca.csv"] = special("static/output/pca.csv", kind, linkname)
    refused(tmp_path, bundle.write(tmp_path / "a.tar"),
            f"member 'static/output/pca.csv/?' is a {label}; a bundle holds regular files only")
    bundle = Bundle(run["bundle"])
    bundle.members = {"static": special("static", kind, linkname), **bundle.members}
    refused(tmp_path, bundle.write(tmp_path / "b.tar"), f"member 'static/?' is a {label}")


def test_symbolic_link_followed_by_a_file_through_it_is_refused(run, tmp_path):
    """The classic escape: a link ``static/output`` to a folder outside, then a file ``static/output/x``."""
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle = Bundle(run["bundle"])
    bundle.members = {"static/landscape": special("static/landscape", tarfile.SYMTYPE, str(outside)), **bundle.members}
    refused(tmp_path, bundle.write(tmp_path / "a.tar"), "member 'static/landscape' is a symbolic link")
    assert list(outside.iterdir()) == []


def test_extended_and_long_name_headers_are_refused(run, tmp_path):
    bundle = Bundle(run["bundle"])
    refused(tmp_path, bundle.write(tmp_path / "pax.tar", first=("x" * 150, b"long name"), fmt=tarfile.PAX_FORMAT),
            "is a extended header; a bundle holds regular files only")
    refused(tmp_path, bundle.write(tmp_path / "gnu.tar", first=("x" * 150, b"long name"), fmt=tarfile.GNU_FORMAT),
            "is a GNU long name; a bundle holds regular files only")
    with tarfile.open(tmp_path / "global.tar", "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "x"}) as tar:
        info = tarfile.TarInfo("bundle.json")
        tar.addfile(info, io.BytesIO(b""))
    refused(tmp_path, tmp_path / "global.tar", "is a global extended header")


def test_sizes_are_capped_before_anything_is_read(run, tmp_path):
    size = run["bundle"].stat().st_size
    trajectory = json.loads(tarfile.open(run["bundle"]).extractfile("bundle.json").read())["files"]["archive/trajectory.xtc"]
    refused(tmp_path, run["bundle"], f"member archive/trajectory.xtc has {trajectory['size']} bytes, at most 1000 are accepted",
            max_member_bytes=1000)
    refused(tmp_path, run["bundle"], r"the members have \d+ bytes together, at most 100000 are accepted", max_bytes=100000)
    assert entry_bundle.check_bundle(run["bundle"], max_bytes=size, max_member_bytes=trajectory["size"])

    # a member that claims to be huge: refused from bundle.json, or from its header, without reading it
    bundle = Bundle(run["bundle"])
    bundle.description["files"]["archive/trajectory.xtc"]["size"] = 1 << 40
    refused(tmp_path, bundle.write(tmp_path / "a.tar"), "member archive/trajectory.xtc has 1099511627776 bytes, at most")
    bundle = Bundle(run["bundle"])
    bundle.members["archive/trajectory.xtc"] = special("archive/trajectory.xtc", tarfile.REGTYPE, size=1 << 33)
    with tarfile.open(tmp_path / "b.tar", "w", format=tarfile.GNU_FORMAT) as tar:
        payload = (json.dumps(bundle.description)).encode()
        info = tarfile.TarInfo("bundle.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        tar.fileobj.write(bundle.members["archive/trajectory.xtc"].tobuf(tarfile.GNU_FORMAT))   # header only, no data
    refused(tmp_path, tmp_path / "b.tar", f"member archive/trajectory.xtc has 8589934592 bytes, bundle.json says {trajectory['size']}")
    # more or fewer bytes than listed
    bundle = Bundle(run["bundle"])
    bundle.members["run.log"] += b"one more line\n"
    refused(tmp_path, bundle.write(tmp_path / "c.tar"), r"member run.log has \d+ bytes, bundle.json says \d+")
    # a bundle.json that is too large to be one
    bundle = Bundle(run["bundle"])
    bundle.description["source"]["comments"] = "x" * (9 << 20)
    refused(tmp_path, bundle.write(tmp_path / "d.tar"), "bundle.json has 94\\d+ bytes, at most 8388608 are accepted")
    bundle.description["source"]["comments"] = "x" * 5000
    refused(tmp_path, bundle.write(tmp_path / "e.tar"), "source.comments is not a text of at most 4000 characters")
    bundle = Bundle(run["bundle"])
    for index in range(5000):
        bundle.description["files"][f"static/f{index}"] = {"size": 0, "sha256": sha256(b"")}
    refused(tmp_path, bundle.write(tmp_path / "f.tar"), "bundle.json lists 50\\d\\d members, at most 4096 are accepted")


def test_name_mismatch(run, tmp_path):
    other = "DGlcpa1-OH"
    bundle = Bundle(run["bundle"])
    bundle.description["glycam_name"] = other
    refused(tmp_path, bundle.write(tmp_path / "a.tar"),
            "archive/manifest.json is the entry 'DManpa1-2DManpa1-OH', bundle.json says 'DGlcpa1-OH'")
    # the manifest renamed (its hash in bundle.json and in the result made to fit): the result is another glycan's
    bundle = Bundle(run["bundle"])
    bundle.description["glycam_name"] = other
    bundle.set_json("archive/manifest.json", dict(bundle.json("archive/manifest.json"), glycam_name=other))
    refused(tmp_path, bundle.refresh_result().refresh().write(tmp_path / "b.tar"),
            "result.json is the result of 'DManpa1-2DManpa1-OH', bundle.json says 'DGlcpa1-OH'")
    # the result renamed alone
    bundle = Bundle(run["bundle"])
    bundle.set_json("result.json", dict(bundle.json("result.json"), glycam_name=other))
    refused(tmp_path, bundle.refresh().write(tmp_path / "c.tar"), "result.json is the result of 'DGlcpa1-OH'")
    # all three renamed consistently: a bundle of that name, which is not the one that was expected
    bundle = Bundle(run["bundle"])
    bundle.description["glycam_name"] = other
    bundle.set_json("archive/manifest.json", dict(bundle.json("archive/manifest.json"), glycam_name=other))
    result = bundle.json("result.json")
    result["glycam_name"] = result["input"]["glycam_name"] = other
    bundle.set_json("result.json", result)
    forged = bundle.refresh_result().refresh().write(tmp_path / "d.tar")
    assert entry_bundle.check_bundle(forged)["glycam_name"] == other
    refused(tmp_path, forged, "the bundle is 'DGlcpa1-OH', 'DManpa1-2DManpa1-OH' was expected",
            expected_glycam_name=GLYCAM_NAME)
    for name, message in (("", "GLYCAM name is missing or empty"), ("a/../b", "not usable as an object key"),
                          (" padded", "surrounding whitespace or control characters"), (7, "missing or empty")):
        bundle = Bundle(run["bundle"])
        bundle.description["glycam_name"] = name
        refused(tmp_path, bundle.write(tmp_path / "e.tar"), message)


def test_archive_files_have_to_be_those_of_the_manifest_and_of_the_result(run, tmp_path):
    # another structure, listed with its own hash: not the file of the manifest
    bundle = Bundle(run["bundle"])
    bundle.members["archive/structure.mol2"] += b"\n"
    refused(tmp_path, bundle.refresh().write(tmp_path / "a.tar"),
            r"archive/structure.mol2 is not the file archive/manifest.json describes \(size or SHA-256\)")
    # ... the manifest made to fit: not the manifest the result was computed from
    bundle = Bundle(run["bundle"])
    bundle.members["archive/structure.mol2"] += b"\n"
    manifest = bundle.json("archive/manifest.json")
    data = bundle.members["archive/structure.mol2"]
    manifest["files"]["structure.mol2"] = {"size": len(data), "sha256": sha256(data)}
    bundle.set_json("archive/manifest.json", manifest)
    refused(tmp_path, bundle.refresh().write(tmp_path / "b.tar"),
            "result.json was computed from another manifest.json than the one in the bundle")
    # another static file than the result records
    bundle = Bundle(run["bundle"])
    bundle.members["static/output/info.json"] = b"{}"
    refused(tmp_path, bundle.refresh().write(tmp_path / "c.tar"),
            "the static files of the bundle are not those result.json records")
    for change, message in (
        (lambda result: result.update(status="failed"), "result.json has status 'failed', not ok"),
        (lambda result: result.update(schema="other"), "result.json is not a glycoshape-entry-result result"),
        (lambda result: result["settings"].update(cluster_levels=[1, 2, 3]), "was computed with levels"),
        (lambda result: result["input"].update(kind="pdb"), "was not computed from an archive entry"),
        (lambda result: result["input"].update(frames=479), "disagree on frames and atoms"),
        (lambda result: result.pop("produced"), "incomplete manifest.json or result.json: KeyError"),
        (lambda result: result["input"]["files"]["trajectory.xtc"].update(sha256="0" * 64), "computed from other files"),
    ):
        bundle = Bundle(run["bundle"])
        result = bundle.json("result.json")
        change(result)
        bundle.set_json("result.json", result)
        refused(tmp_path, bundle.refresh().write(tmp_path / "d.tar"), message)
    bundle = Bundle(run["bundle"])
    bundle.set_json("archive/manifest.json", dict(bundle.json("archive/manifest.json"), schema="something"))
    refused(tmp_path, bundle.refresh().write(tmp_path / "e.tar"), "is not a glycoshape-raw-trajectory manifest")
    bundle = Bundle(run["bundle"])
    bundle.members["result.json"] = b"[1, 2]"
    refused(tmp_path, bundle.refresh().write(tmp_path / "f.tar"), "result.json is not a JSON object")


def test_description_has_to_be_what_this_code_reads(run, tmp_path):
    def with_description(change):
        bundle = Bundle(run["bundle"])
        change(bundle.description)
        return bundle.write(tmp_path / "a.tar")

    for change, message in (
        (lambda d: d.update(schema="glycoshape-raw-trajectory"), "bundle.json is not a glycoshape-entry-bundle description"),
        (lambda d: d.update(schema_version=2), "has schema version 2, this code reads version 1"),
        (lambda d: d.update(schema_version=True), "has schema version True"),
        (lambda d: d.update(levels=[1, 3]), r"levels must be \[1, 2\] or \[1, 2, 3\]"),
        (lambda d: d.update(levels=[1, 2, 3]), "bundle.json does not list static/output/level_3/dist.svg"),
        (lambda d: d.pop("extends"), "extends is missing"),
        (lambda d: d.update(extends={"glycam_name": GLYCAM_NAME, "manifest_sha256": "0" * 64, "frames": 100}),
         "disagree on whether an entry is extended"),
        (lambda d: d.update(extends={"glycam_name": GLYCAM_NAME, "manifest_sha256": "zz", "frames": 100}),
         "extends.manifest_sha256 is not a SHA-256"),
        (lambda d: d.update(extends={"glycam_name": "DGlcpa1-OH", "manifest_sha256": "0" * 64, "frames": 100}),
         "the bundle is DManpa1-2DManpa1-OH and extends DGlcpa1-OH"),
        (lambda d: d.update(files=[]), "files is missing or empty"),
        (lambda d: d["files"]["run.log"].update(sha256="ABC"), "needs a size and a SHA-256"),
        (lambda d: d["files"]["run.log"].update(size=-1), "needs a size and a SHA-256"),
        (lambda d: d["files"]["run.log"].update(size=1.5), "needs a size and a SHA-256"),
        (lambda d: d.update(files=dict(reversed(list(d["files"].items())))), "files is not in the order of a bundle"),
        (lambda d: d["files"].update({"static/Output/pca.csv": d["files"]["static/output/pca.csv"]}),
         "differ only in the case of their letters"),
        (lambda d: d["files"].update({"static/output/pca.csv/x": d["files"]["static/output/pca.csv"]}),
         "static/output/pca.csv is listed as a file and as a folder"),
        (lambda d: d.update(source=None), "source is missing"),
        (lambda d: d["source"].update(temperature_k="300"), "source.temperature_k is '300', not a number"),
        (lambda d: d.update(pipeline="v1"), "pipeline is missing"),
        (lambda d: d.update(created=20261011), "created is not a time stamp"),
    ):
        refused(tmp_path, with_description(change), message)

    bundle = Bundle(run["bundle"])
    text = json.dumps(bundle.description)
    for payload, message in (
        (b"not json", "bundle.json is not readable JSON"),
        (b"\xff\xfe", "bundle.json is not readable JSON"),
        (b"[]", "bundle.json is not a JSON object"),
        (text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1').encode(), "occurs twice"),
        (text.replace('"temperature_k": 300.0', '"temperature_k": NaN').encode(), "NaN is not a JSON number"),
        (b"[" * 100000, "bundle.json is not readable JSON"),
    ):
        refused(tmp_path, bundle.write(tmp_path / "b.tar", description=payload), message)


def test_files_that_are_not_a_bundle(run, tmp_path):
    data = run["bundle"].read_bytes()
    (tmp_path / "bundle.tar.gz").write_bytes(gzip.compress(data))
    refused(tmp_path, tmp_path / "bundle.tar.gz", "is gzip-compressed; a bundle is a plain tar file")
    (tmp_path / "cut.tar").write_bytes(data[: len(data) // 2])
    refused(tmp_path, tmp_path / "cut.tar", r"cut.tar ends inside \S+ \(at byte \d+\)")
    (tmp_path / "no-end.tar").write_bytes(data.rstrip(b"\0"))
    refused(tmp_path, tmp_path / "no-end.tar", "ends inside")
    (tmp_path / "tail.tar").write_bytes(data + b"appended")
    refused(tmp_path, tmp_path / "tail.tar", "has data after the end of the archive")
    (tmp_path / "padded.tar").write_bytes(data + bytes(2 << 20))
    refused(tmp_path, tmp_path / "padded.tar", "has data after the end of the archive")
    (tmp_path / "empty.tar").write_bytes(b"")
    refused(tmp_path, tmp_path / "empty.tar", "ends inside a tar header")
    (tmp_path / "zeros.tar").write_bytes(bytes(10240))
    refused(tmp_path, tmp_path / "zeros.tar", "the first member is 'missing', not bundle.json")
    (tmp_path / "text.tar").write_bytes(b"this is not a tar file\n" * 100)
    refused(tmp_path, tmp_path / "text.tar", "not a tar header at byte 0")
    damaged = bytearray(data)
    damaged[0] ^= 0x01                                        # the name in the first header: its checksum is off
    (tmp_path / "header.tar").write_bytes(bytes(damaged))
    refused(tmp_path, tmp_path / "header.tar", "not a tar header at byte 0: the checksum is wrong")

    bundle = Bundle(run["bundle"])
    refused(tmp_path, bundle.write(tmp_path / "first.tar", first=("run.log", b"first")),
            "the first member is 'run.log', not bundle.json")
    assert entry_bundle.open_bundle(run["bundle"], tmp_path / "good") == tmp_path / "good"     # the unchanged one passes


def test_bundle_of_an_extended_entry_has_to_agree_with_its_manifest(run, tmp_path):
    bundle = Bundle(run["bundle"])
    manifest = bundle.json("archive/manifest.json")
    extends = {"glycam_name": GLYCAM_NAME, "manifest_sha256": "1" * 64, "frames": 100}
    manifest["extends"] = dict(extends, atoms=45)
    bundle.set_json("archive/manifest.json", manifest)
    bundle.refresh_result().refresh()
    refused(tmp_path, bundle.write(tmp_path / "a.tar"), "disagree on whether an entry is extended")
    for change, message in (
        ({}, None),
        ({"frames": 99}, "disagree on extends.frames"),
        ({"manifest_sha256": "2" * 64}, "disagree on extends.manifest_sha256"),
        ({"frames": 0}, "extends.frames is not a number of frames"),
    ):
        bundle.description["extends"] = dict(extends, **change)
        written = bundle.write(tmp_path / "b.tar")
        if message is None:
            assert entry_bundle.check_bundle(written)["extends"] == extends
        else:
            refused(tmp_path, written, message)
    manifest["extends"]["frames"] = extends["frames"] = 480
    bundle.set_json("archive/manifest.json", manifest)
    bundle.description["extends"] = extends
    refused(tmp_path, bundle.refresh_result().refresh().write(tmp_path / "c.tar"),
            "the entry has 480 frames and says it extends 480")
