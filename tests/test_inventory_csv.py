"""The submission inventory CSV: empty cells, and the published file without e-mail addresses."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

import lib.config as config
from lib import glystatic, name_utils
from lib import storage as storage_module
from lib.storage import LocalStorageBackend, StorageManager

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "GlycoShape_Inventory.csv"
HEADER = ["ID", "Timestamp", "Email address", "Full GLYCAM name of glycan being submitted.",
          "How will the data be transferred?", "What is the aggregated length of the simulations?",
          "What MD package was used for the simulations?", "What force field was used for the simulations?",
          "What temperature target was used for the simulations? ", "What pressure target was used for the simulations?",
          "What NaCl concentration was used for the simulations?",
          "Any comments that should be noted with the submission?", "What is the GlyTouCan ID of the glycan?"]


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    """Write an inventory CSV (with or without the e-mail column) and point the pipeline at it."""
    monkeypatch.setattr(storage_module, "_storage_manager", StorageManager(LocalStorageBackend()))
    monkeypatch.setattr(config, "use_pocketbase", False)
    if hasattr(name_utils.get_md_info, "_inventory_df"):
        monkeypatch.delattr(name_utils.get_md_info, "_inventory_df")

    def write(rows, email_column=True):
        header = HEADER if email_column else [h for h in HEADER if h != "Email address"]
        path = tmp_path / "GlycoShape_Inventory.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row if email_column else row[:2] + row[3:])
        monkeypatch.setattr(config, "inventory_path", path)
        return path
    return write


ROW_WITHOUT_ACCESSION = ["GS00445", "27/11/2023 12:10:00", "someone@example.org", "DManpa1-2DManpa1-OH", "copied", "1.5",
                         "AMBER", "GLYCAM_06j", "300.0", "1.0", "200", "", ""]


@pytest.mark.parametrize("email_column", [True, False])
def test_empty_glytoucan_cell_is_no_accession(inventory, email_column):
    inventory([ROW_WITHOUT_ACCESSION, ["GS00446", "22/12/2023 06:19:50", "", "DGlcpb1-OH", "copied", "1.5", "GROMACS",
                                       "GLYCAM_06j", "300.0", "1.0", "150", "Cork", "TODO"]], email_column)
    metadata = glystatic._get_metadata_from_csv("DManpa1-2DManpa1-OH")
    assert metadata["glytoucan_id"] == "" and metadata["comments"] is None
    assert metadata["ID"] == "GS00445" and metadata["package"] == "AMBER" and float(metadata["length"]) == 1.5
    assert metadata["email"] == ("someone@example.org" if email_column else None)
    # what the static stage makes of it: no archetype accession, not "NaN"
    assert glystatic._apply_glytoucan_fallbacks(None, None, None, metadata["glytoucan_id"]) == (None, None, None)
    assert glystatic._get_metadata_from_csv("DGlcpb1-OH")["glytoucan_id"] == ""


def test_accession_in_the_inventory_is_kept(inventory):
    inventory([ROW_WITHOUT_ACCESSION[:-1] + [" G12345AB "]])
    metadata = glystatic._get_metadata_from_csv("DManpa1-2DManpa1-OH")
    assert metadata["glytoucan_id"] == "G12345AB"
    assert glystatic._apply_glytoucan_fallbacks(None, None, None, metadata["glytoucan_id"])[0] == "G12345AB"


@pytest.mark.parametrize("email_column", [True, False])
def test_md_info_reads_columns_by_name(inventory, email_column):
    inventory([ROW_WITHOUT_ACCESSION], email_column)
    assert name_utils.get_md_info("DManpa1-2DManpa1-OH") == (
        "GS00445", "1.5", "AMBER", "GLYCAM_06j", "300.0", "1.0", 200, "someone@example.org" if email_column else None)


def test_published_inventory_has_no_email_addresses(inventory, monkeypatch):
    text = PUBLISHED.read_text()
    assert "Email address" not in text.splitlines()[0]
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    # and the pipeline still reads it
    monkeypatch.setattr(config, "inventory_path", PUBLISHED)
    metadata = glystatic._get_metadata_from_csv("DManpa1-2DManpa1-OH")
    assert (metadata["ID"], metadata["package"], metadata["glytoucan_id"], metadata["email"]) == ("GS00445", "AMBER", "", None)
    assert name_utils.get_md_info("DManpa1-2DManpa1-OH")[:3] == ("GS00445", "1.5", "AMBER")
