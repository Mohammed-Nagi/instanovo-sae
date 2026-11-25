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


# --- Candidates, selection, determined ions (annotate.py) -------------------------

def _fa(ion_type, charge=1, neutral_loss=None, isotope=0, ppm=1.0):
    from types import SimpleNamespace
    return SimpleNamespace(ion_type=ion_type, charge=charge, neutral_loss=neutral_loss,
                           isotope=isotope, mz_delta=(ppm, "ppm"))


def test_annotation_to_dict_is_first_candidate():
    from types import SimpleNamespace
    # internal without a loss beats b with a loss (loss is ranked above ion class)
    pi = SimpleNamespace(fragment_annotations=[_fa("b3", neutral_loss="-H2O"), _fa("m2:4", ppm=-3.0),
                                               _fa("y2", isotope=1)])
    cands = annotate._annotation_candidates(pi)
    assert [c["ion_type"] for c in cands] == ["internal", "b", "y"]
    assert annotate._annotation_to_dict(pi) == {"ion_type": "internal", "ion_index": (2, 4), "charge": 1,
                                                "neutral_loss": ""}
    assert annotate._annotation_to_dict(SimpleNamespace(fragment_annotations=[]))["ion_type"] == "noise"


def test_select_annotation_settings():
    c = [{"ion_type": "a", "ion_index": 2, "charge": 1, "neutral_loss": "", "isotope": 0, "mz_delta": 0.0},
         {"ion_type": "internal", "ion_index": (2, 4), "charge": 1, "neutral_loss": "", "isotope": 0,
          "mz_delta": 0.0},
         {"ion_type": "y", "ion_index": 3, "charge": 1, "neutral_loss": "", "isotope": 1, "mz_delta": 0.0}]
    assert annotate.select_annotation(c)["ion_type"] == "internal"            # pipeline setting
    assert annotate.select_annotation(c, 0, ())["ion_type"] == "a"           # a ions on
    assert annotate.select_annotation(c[2:])["ion_type"] == "noise"          # isotope-only peak
    assert annotate.select_annotation(c[2:], 1)["ion_index"] == 3


def test_run_spectrum_utils_candidates_reproduce_labels():
    import torch
    proton = 1.007276
    b2 = 97.052764 + 129.042593 + proton
    y2 = 129.042593 + 128.094963 + 18.010565 + proton
    mz = torch.tensor([b2 - 18.010565, b2, y2, y2 + 1.003355, 500.0], dtype=torch.float64)
    kw = dict(proforma="PEPTIDEK", mz_array=mz, intensity_array=torch.ones(5), precursor_mz=464.7347,
              precursor_charge=2, ion_types="byIp", fragment_tol_mass=20.0, fragment_tol_mode="ppm",
              enable_internal=True, strict=True)
    labels = annotate.run_spectrum_utils(**kw)
    cands = annotate.run_spectrum_utils(**(kw | {"ion_types": "abyIp", "max_isotope": 1}),
                                        return_candidates=True)
    derived = [annotate.select_annotation(c) for c in cands]
    assert [{k: d[k] for k in annotate._LABEL_KEYS} for d in derived] == labels
    assert annotate.select_annotation(cands[3], 1)["isotope"] == 1 and labels[3]["ion_type"] == "noise"


def test_is_composition_determined():
    det = annotate.is_composition_determined
    assert det({"ion_type": "I", "ion_index": "W"}, 8) and det({"ion_type": "precursor", "ion_index": None}, 8)
    assert det({"ion_type": "y", "ion_index": 1}, 8) and det({"ion_type": "b", "ion_index": 7}, 8)
    assert det({"ion_type": "a", "ion_index": 7}, 8)
    assert not det({"ion_type": "y", "ion_index": 7}, 8) and not det({"ion_type": "b", "ion_index": 1}, 8)
    assert not det({"ion_type": "internal", "ion_index": (2, 7)}, 8)


def test_annotation_inputs_from_tokens_moves_modifications():
    mods = [{"position": 3, "mod_name": "+0.98", "unimod_id": -1},
            {"position": 5, "mod_name": "+15.99", "unimod_id": 35}]
    bare, prof, out = annotate.annotation_inputs_from_tokens(["M[+15.99]", "A", "N[+0.98]", "G", "K"], mods)
    assert (bare, prof) == ("MANGK", "M[+15.99]AN[+0.98]GK")
    assert sorted((m["position"], m["mod_name"], m["unimod_id"]) for m in out) == \
        [(1, "+15.99", 35), (3, "+0.98", -1)]
    with pytest.raises(ValueError):
        annotate.annotation_inputs_from_tokens(["A", "N[+0.98]", "K"], [])


def test_peptide_decoys_shared_with_runtime():
    import numpy as np
    from inspector import runtime
    assert runtime.peptide_decoys is annotate.peptide_decoys
    pep = list("PEPTIDEK")
    a = annotate.peptide_decoys(pep, 5, np.random.default_rng([0, 7]), exclude=[pep])
    b = annotate.peptide_decoys(pep, 5, np.random.default_rng([0, 7]), exclude=[pep])
    assert a == b and all(d[-1] == "K" for d in a)


def test_spectrum_utils_internal_end_is_exclusive():
    # Finding 9: m2:4 of PEPTIDEK is residues 2-3 (EP), but IonGeometry treats the end as inclusive
    import spectrum_utils.fragment_annotation as fa
    import spectrum_utils.proforma as pf
    frags = {f.ion_type: m for f, m in fa.get_theoretical_fragments(pf.parse("PEPTIDEK")[0], "m", max_charge=1)}
    assert frags["m2:4"] == pytest.approx(129.042593 + 97.052764 + 1.007276, abs=1e-5)
    assert annotate.IonGeometry("internal", (2, 4), 8).covers(4)          # the extra residue


def test_spectrum_utils_immonium_ignores_the_peptide():
    import spectrum_utils.fragment_annotation as fa
    import spectrum_utils.proforma as pf
    labels = {f.ion_type for f, _m in fa.get_theoretical_fragments(pf.parse("PEPTIDEK")[0], "I")}
    assert {"IW", "IY", "IR"} <= labels                                    # no W, Y or R in PEPTIDEK


# --- Audit aggregation (analysis.py) ---------------------------------------------

def test_allocate_sample():
    import numpy as np
    k = analysis.allocate_sample(13, 200, np.random.default_rng(0))
    assert sum(k) == 200 and sorted(set(k)) == [15, 16] and k.count(16) == 5


def test_ratio_cis():
    import numpy as np
    idx = analysis.bootstrap_indices(4, 50, 0)
    est, lo, hi = analysis.ratio_ci([1, 2, 3, 4], [2, 4, 6, 8], idx)
    assert est == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    est, lo, hi = analysis.ratio_diff_ci([1, 1], [2, 2], [1, 1], [4, 4], analysis.bootstrap_indices(2, 50, 0))
    assert est == pytest.approx(0.25) and lo == pytest.approx(0.25) and hi == pytest.approx(0.25)
    assert np.isnan(analysis.ratio_ci([0, 0], [0, 0], idx[:, :2] % 2)[0])


def test_concept_label_classes_cover_the_registry():
    assert set(analysis.CONCEPT_LABEL_CLASSES) == set(annotate.ConceptRegistry.build().names)


def _cand(t, k, loss="", charge=1, iso=0):
    return {"ion_type": t, "ion_index": k, "charge": charge, "neutral_loss": loss, "isotope": iso,
            "mz_delta": 0.0}


def test_peptide_label_metrics_toy():
    import torch
    pep = "PEPDIDEK"                                   # P1 E2 P3 D4 I5 D6 E7 K8
    cands = [
        [_cand("internal", (2, 4)), _cand("b", 3, "H2O")],   # sorted best first: internal beats b-H2O
        [],                                                   # noise
        [_cand("y", 3, iso=1)],                               # noise; 13C peak of y3
        [_cand("y", 3)],                                      # y3 itself
        [_cand("a", 2)],                                      # noise; a2 when a ions are on
        [_cand("I", "W")],                                    # immonium of an absent residue
        [_cand("y", 1)],                                      # composition-determined
        [_cand("b", 2)],
    ]
    mz = torch.tensor([200.0, 250.0, 400.0, 401.0, 500.0, 159.09, 147.11, 227.1], dtype=torch.float64)
    m = analysis.peptide_label_metrics(cands, mz, torch.ones(8), pep, [], 2, annotate.ConceptRegistry.build())
    assert m["cls|internal|none|1"] == 1 and m["cls|y|*|*"] == 1 and m["cls|b|*|*"] == 1
    assert m["cls|a|none|1"] == 1 and m["det|y1"] == 1 and "det|immonium" in m
    assert m["prec|bcand_win|internal"] == 1 and m["prec|bcand_reason|loss"] == 1
    assert m["iso|noise|all"] == 3 and m["iso|became|all"] == 1 and m["iso|became_m0|all"] == 1
    assert m["aion|from|noise|a2"] == 1 and m["aion|den|noise"] == 3
    assert m["con|intextra|covers_D"] == 1                    # residue 4 (D) is the internal ion's extra
    assert m["imm|absent"] == 1 and m["con|imm_absent|covers_W"] == 1
    assert m["con|diso|is_y_ion"] == 1 and m["con|da|is_noise_peak"] == -1
    assert m["cov|b|1|none|all|num"] == 1 and m["cov|b|1|none|all|den"] == 7
    assert m["cov|y|1|any|chance|den"] == 6 and m["cov|y|1|any|chance|num"] == 1   # y1 excluded
