"""Toy-input tests for the label-quality audit (Finding 9) and the deamidation
precursor check (Finding 8(d)): pure functions in annotate.py and analysis.py.
No local data needed."""
from __future__ import annotations

import pytest

import analysis
import annotate

# PEPTIDEK residues (pyteomics monoisotopic masses), summed by hand.
PEPTIDEK_RESIDUES = 2 * 97.052764 + 2 * 129.042593 + 101.047679 + 113.084064 + 115.026943 + 128.094963


# --- ProForma tokens and masses ------------------------------------------------

def test_proforma_residue_tokens_round_trip():
    for s in ("PEPTIDEK", "DTFN[+0.98]TSM[+15.99]K", "[+42.01]-PEPK", "C[+57.02]AK"):
        toks = annotate.proforma_residue_tokens(s)
        assert "".join(toks) == s
    assert annotate.proforma_residue_tokens("PEN[+0.98]K") == ["P", "E", "N[+0.98]", "K"]
    assert annotate.proforma_residue_tokens("[+42.01]-PEK")[0] == "[+42.01]-"
    for bad in ("PEPk", "PEP[+1]-[+2]K", "PE[+1"):
        with pytest.raises(ValueError):
            annotate.proforma_residue_tokens(bad)


def test_proforma_mass_and_mz_by_hand():
    assert annotate.proforma_neutral_mass("PEPTIDEK") == pytest.approx(PEPTIDEK_RESIDUES + 18.010565, abs=1e-5)
    # the dataset's rounded +0.98 is replaced by the exact deamidation mass
    assert annotate.proforma_neutral_mass("PEPTIN[+0.98]K") - annotate.proforma_neutral_mass("PEPTINK") \
        == pytest.approx(0.984016, abs=1e-9)
    m = PEPTIDEK_RESIDUES + 18.010565
    assert annotate.proforma_mz("PEPTIDEK", 2) == pytest.approx((m + 2 * 1.007276) / 2, abs=1e-5)


def test_precursor_ppm_matches_instanovo_definition():
    assert annotate.precursor_ppm(1000.0, 1000.01, 1, 0) == pytest.approx(-0.01 / 1000.01 * 1e6)
    # measured = first 13C peak of the theoretical ion -> 0 ppm at offset 1
    assert annotate.precursor_ppm(500.0, 500.0 + 1.00335 / 2, 2, 1) == pytest.approx(0.0, abs=1e-9)


# --- Deamidation vs 13C ---------------------------------------------------------

def test_remove_one_deamidation():
    assert analysis.remove_one_deamidation("PEN[+0.98]KQ[+0.98]R") == "PENKQ[+0.98]R"
    assert analysis.remove_one_deamidation("PEM[+15.99]K") is None


def test_precursor_explanations_13c_artefact():
    # observed = 13C peak of the amidated peptide; the label calls it deamidated
    amid = "AAAAAAAAAANK"
    z = 2
    observed = annotate.proforma_mz(amid, z) + 1.00335 / z
    e = analysis.precursor_explanations("AAAAAAAAAAN[+0.98]K", observed, z)
    assert e["ppm_amidated_iso1"] == pytest.approx(0.0, abs=1e-6)
    gap = (0.984016 - 1.00335) / z / observed * 1e6        # label sits 19.3 mDa below the observed mass
    assert e["ppm_label_iso0"] == pytest.approx(gap, rel=1e-6)
    assert e["amidated_meets_iso1"] and not e["label_meets_only_iso1"]
    assert analysis.deamidation_d(e["ppm_label_iso0"], e["ppm_amidated_iso1"], 0.0) == pytest.approx(abs(gap))
    assert "ppm_amidated_iso1" not in analysis.precursor_explanations("PEPTIDEK", 465.0, 2)


def test_mass_bin_edges():
    assert [analysis.mass_bin(m) for m in (500, 967, 1499.9, 1500, 3000)] == \
        ["<967", "967-1500", "967-1500", "1500-2500", ">=2500"]


def test_explanation_fit_and_gap():
    assert analysis.explanation_fit(0.5, 15.0, 0.0, 4.0) == "deamidated_only"
    assert analysis.explanation_fit(-15.0, 0.5, 0.0, 4.0) == "13c_only"
    assert analysis.explanation_fit(3.0, -3.0, 0.0, 4.0) == "both"
    assert analysis.explanation_fit(9.0, 9.0, 0.0, 4.0) == "neither"
    assert analysis.gap_ppm(1933.4) == pytest.approx(10.0, rel=1e-3)
