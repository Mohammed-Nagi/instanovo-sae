"""annotate.py — multi-concept token annotation for InstaNovo SAE evaluation.

Reads the v4 per-chunk metadata written by extract.py (the ChunkMeta
files — activations are not needed here) and writes one LabelChunkData per chunk
with the same token ordering, ready for row-by-row joining with activations.

Fragment-ion annotation uses spectrum_utils 0.5.x (mzPAF-style annotations). The
adapter below is the single place that depends on the spectrum_utils annotation
object layout, so a future version bump is localised there.

Mass-spectrometry conventions (the nine-species benchmark is high-resolution
Orbitrap data, so these matter):
  - Fragment matching uses a ppm tolerance (default 20 ppm), not a wide Da window;
    a 0.5 Da window matches many spurious peaks at Orbitrap resolution.
  - Modification masses are resolved to their exact monoisotopic values before
    annotation. The dataset rounds deltas to two decimals (e.g. "+15.99" for
    oxidation, exact 15.994915); at 20 ppm that rounding can push a modified
    fragment out of tolerance, so known PTMs are substituted with exact masses.
  - Only monoisotopic fragments are generated (max_isotope=0); isotope peaks of a
    real fragment fall through to the noise label, which is the conservative choice
    for per-token concept labelling.
  - Immonium ions indicate a residue's PRESENCE, not a position, so they drive only
    is_I_ion and the matching covers_<residue> concept — never position, cleavage,
    or PTM-localisation concepts.

PTM concepts are identified via InstaNovo's canonical LEGACY_PTM_TO_UNIMOD table:
the dataset writes delta-mass modifications, and the table maps each legacy token
(e.g. "M(+15.99)") to its UNIMOD id — the same mapping the model applies to build
its residue vocabulary, so the labels match what the encoder saw. Fragment
annotation, however, keeps exact delta masses rather than UNIMOD tags, because
spectrum_utils resolves UNIMOD ids over the network (see exact_mass_proforma).
The tracked PTMs are oxidation, carbamidomethyl-Cys, and deamidation; mass
matching is retained only as a fallback when the table is unavailable.

Concept families: ion type (b / y / immonium / internal / precursor / noise),
neutral loss, fragment charge, precursor charge, position, mass region, intensity,
first/last ion, canonical vs enhanced cleavage, residue cover, peptide length,
and token- vs spectrum-level PTM containment (the spectrum-level set is a
diagnostic negative control).

Output layout under --output-dir:
    annotation_manifest.json   # registry, base rates, vocabularies
    concept_phi.pt             # concept-concept correlation matrix + counts
    labels/
        chunk_00000.pt         # one LabelChunkData per input chunk
        chunk_00001.pt
        ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# spectrum_utils 0.5.x is the fragment-ion annotation engine. The annotation
# object layout (PeakInterpretation -> FragmentAnnotation) is read in exactly one
# place, _annotation_to_dict, which is where any future-version tweak would go.
import spectrum_utils.spectrum as sus

# InstaNovo's canonical legacy-token -> UNIMOD ProForma table (see module docstring).
# Guarded so this file still runs without the InstaNovo package; PTM identity then
# falls back to mass matching against the tracked PTMs (see _mod_unimod_id).
try:
    from instanovo.constants import LEGACY_PTM_TO_UNIMOD
except Exception:
    LEGACY_PTM_TO_UNIMOD: dict[str, str] = {}

SCHEMA_VERSION = 4          # LabelChunkData schema (row-aligned with v4 activations)
EXTRACT_SCHEMA_VERSION = 4  # extract.py manifest/ChunkMeta this reads
LOG = logging.getLogger("annotate")


# ─────────────────────────────────────────────────────────────────────────────
# Concept registry
# ─────────────────────────────────────────────────────────────────────────────
# Families are ordered for consistent indexing across chunks. Adding new
# concepts at the end of a family preserves backward compatibility.
CONCEPT_FAMILIES: dict[str, list[str]] = {
    "ion_type": [
        "is_b_ion", "is_y_ion", "is_I_ion",
        "is_internal_fragment", "is_precursor_related",
        "is_matched_peak", "is_noise_peak", "is_latent_token",
    ],
    "neutral_loss": ["is_H2O_loss", "is_NH3_loss", "has_neutral_loss"],
    "fragment_charge": ["is_fragment_charge_1", "is_fragment_charge_2"],
    "precursor_charge": ["precursor_charge_2", "precursor_charge_3", "precursor_charge_4"],
    "position": ["position_Nterm", "position_middle", "position_Cterm"],
    "mass_region": ["mz_low", "mz_mid", "mz_high"],
    "intensity": ["top_decile_intensity"],
    "first_last_ion": ["is_first_ion", "is_last_ion"],
    "cleavage_canonical": ["cleaves_C_to_Lys", "cleaves_C_to_Arg"],
    "cleavage_enhanced": ["cleaves_N_to_Pro", "cleaves_C_to_Asp", "cleaves_C_to_Glu"],
    "residue_cover": [
        "covers_K", "covers_R",          # trypsin termini
        "covers_W", "covers_F", "covers_Y",  # aromatic (immonium-prone)
        "covers_P",                          # Pro-effect
        "covers_M",                          # oxidation
        "covers_C",                          # cam-cys
        "covers_D", "covers_E",          # Asp/Glu-effect
        "covers_N",                          # deamidation
    ],
    "peptide_length": ["peptide_length_short", "peptide_length_medium", "peptide_length_long"],
    "ion_ptm": ["ion_contains_oxidation", "ion_contains_cam_cys", "ion_contains_deamidation"],
    "spectrum_ptm_diagnostic": [
        "spectrum_contains_oxidation",
        "spectrum_contains_cam_cys",
        "spectrum_contains_deamidation",
    ],
}

# Diagnostic families are computed and saved but expected to fail causal
# ablation — they exist as negative controls for methodology validation.
DIAGNOSTIC_FAMILIES: set[str] = {"spectrum_ptm_diagnostic"}

# Mass-region bin boundaries in Da.
MZ_LOW_HIGH_CUT = (300.0, 1000.0)

# Peptide-length bin boundaries (inclusive on left).
PEPTIDE_LEN_CUTS = (7, 16)  # short < 7, medium 7..15, long >= 16

# Tracked PTMs, keyed by UNIMOD id -> (concept short name, exact monoisotopic
# mass in Da). These three are the only modifications with non-trivial base rate
# in the nine-species benchmark, and the concept registry tracks exactly these.
# The exact masses drive exact_mass_proforma (fragment matching); the names drive
# the ion_contains_* / spectrum_contains_* concepts. Identity (which UNIMOD id a
# modification is) comes from LEGACY_PTM_TO_UNIMOD, see _mod_unimod_id.
TRACKED_PTMS: dict[int, tuple[str, float]] = {
    35: ("oxidation",   15.994915),   # "+15.99" on M
    4:  ("cam_cys",     57.021464),   # "+57.02" on C (carbamidomethyl)
    7:  ("deamidation",  0.984016),   # "+0.98"/"+.98" on N/Q
}
# Tolerance for the mass-matching fallback (used only when a token is absent from
# LEGACY_PTM_TO_UNIMOD). The dataset rounds to two decimals (~0.005 Da error) and
# the tracked masses are well separated, so 0.02 Da is safe and unambiguous.
PTM_MASS_TOL = 0.02

_MASS_DELTA_RE = re.compile(r"[+-]?\d*\.?\d+")
_UNIMOD_ID_RE = re.compile(r"UNIMOD:(\d+)")

# Small int vocabularies for compact per-token metadata storage.
ION_TYPE_VOCAB: dict[str, int] = {
    "noise": 0, "latent": 1, "precursor": 2,
    "b": 3, "y": 4, "I": 5, "internal": 6,
    "a": 7, "c": 8, "x": 9, "z": 10,
}
ION_TYPE_INV: dict[int, str] = {v: k for k, v in ION_TYPE_VOCAB.items()}

NEUTRAL_LOSS_VOCAB: dict[str, int] = {
    "": 0, "H2O": 1, "NH3": 2, "H2O+NH3": 3, "H3PO4": 4,
}
NEUTRAL_LOSS_INV: dict[int, str] = {v: k for k, v in NEUTRAL_LOSS_VOCAB.items()}


@dataclasses.dataclass
class ConceptRegistry:
    """The full set of concepts, with family memberships and a name→index map."""

    names: list[str]
    family_of: dict[str, str]
    diagnostic: set[str]

    @classmethod
    def build(cls) -> ConceptRegistry:
        names: list[str] = []
        family_of: dict[str, str] = {}
        diagnostic: set[str] = set()
        for family, concepts in CONCEPT_FAMILIES.items():
            for c in concepts:
                names.append(c)
                family_of[c] = family
                if family in DIAGNOSTIC_FAMILIES:
                    diagnostic.add(c)
        return cls(names=names, family_of=family_of, diagnostic=diagnostic)

    @property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @property
    def ptm_keys(self) -> list[str]:
        return sorted(name for name, _mass in TRACKED_PTMS.values())

    def to_jsonable(self) -> dict:
        return {
            "names": self.names,
            "family_of": self.family_of,
            "diagnostic": sorted(self.diagnostic),
            "ptm_keys": self.ptm_keys,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Chunk schema
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class LabelChunkData:
    """Per-token labels for one input chunk. Same token ordering and total_tokens
    as the corresponding extraction ChunkMeta, so the evaluator can join these
    labels row-for-row against activations[layer] without index translation.
    """

    schema_version: int
    chunk_idx: int
    n_spectra: int
    total_tokens: int

    # Concept names for self-description; full registry lives in the manifest.
    concept_names: list[str]

    # Main payload: per-token boolean labels [total_tokens, n_concepts].
    # torch.bool gives ~6x memory saving over float; downstream code casts as needed.
    token_labels: torch.Tensor

    # Per-token metadata for analysis tools (filter without re-derivation).
    ion_type_ids: torch.Tensor       # [total_tokens] int8, lookup via ION_TYPE_INV
    ion_indices: torch.Tensor        # [total_tokens] int16, -1 if N/A
    fragment_charges: torch.Tensor   # [total_tokens] int8
    neutral_loss_ids: torch.Tensor   # [total_tokens] int8, lookup via NEUTRAL_LOSS_INV
    peak_mzs: torch.Tensor           # [total_tokens] float32, 0 for latent
    peak_intensities: torch.Tensor   # [total_tokens] float32, 0 for latent

    def save(self, path: Path) -> None:
        torch.save(dataclasses.asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> LabelChunkData:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


# ─────────────────────────────────────────────────────────────────────────────
# Ion geometry — what residue positions does each fragment ion cover?
# ─────────────────────────────────────────────────────────────────────────────
class IonGeometry:
    """Compute geometric coverage of a fragment ion over its parent peptide.

    The 1-indexed residue positions covered by an ion of given type and index
    are what enable token-level PTM and residue-cover labels. Immonium ions are a
    special case: they indicate a residue's presence anywhere in the peptide, not
    a position, so they cover their residue type only (never a position, which
    means they cannot localise a PTM).
    """

    __slots__ = ("ion_type", "ion_index", "peptide_length", "internal_end", "residue")

    def __init__(
        self,
        ion_type: str,
        ion_index,  # int (backbone) | tuple[int, int] (internal) | str (immonium residue) | None
        peptide_length: int,
    ):
        self.ion_type = ion_type
        self.peptide_length = peptide_length
        self.residue = None
        self.internal_end = -1
        if isinstance(ion_index, tuple):
            self.ion_index, self.internal_end = ion_index
        elif isinstance(ion_index, str):     # immonium: index carries the residue
            self.residue = ion_index
            self.ion_index = -1
        else:
            self.ion_index = ion_index if ion_index is not None else -1

    def covers(self, position_1indexed: int) -> bool:
        """Does this ion span the given (1-indexed) residue position?"""
        p, L = position_1indexed, self.peptide_length
        t = self.ion_type
        k = self.ion_index
        if t == "b":
            return 1 <= p <= k
        if t == "y":
            return L - k + 1 <= p <= L
        if t == "internal":
            return self.ion_index <= p <= self.internal_end
        if t in ("precursor", "latent"):
            return 1 <= p <= L
        # Immonium ("I") covers no specific position; noise/unknown cover nothing.
        return False

    def covers_any_residue(self, peptide: str, residue: str) -> bool:
        """Does this ion cover at least one occurrence of `residue` in `peptide`?"""
        if self.ion_type == "I":
            # An immonium ion exists only because its residue is present, so it
            # attests that residue type — but not where.
            return residue == self.residue
        for i, aa in enumerate(peptide, start=1):
            if aa == residue and self.covers(i):
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Modification helpers (UNIMOD identity via the canonical table + exact-mass ProForma)
# ─────────────────────────────────────────────────────────────────────────────
def _mod_delta_mass(mod: dict) -> float | None:
    """Parse the numeric delta mass from a modification's mod_name, or None if it
    is a UNIMOD tag (handled separately) or unparseable."""
    name = str(mod.get("mod_name", ""))
    if "UNIMOD" in name.upper():
        return None
    m = _MASS_DELTA_RE.search(name)
    return float(m.group(0)) if m else None


def _mod_unimod_id(mod: dict, peptide: str) -> int:
    """Resolve a modification to its UNIMOD id.

    Identity is taken from InstaNovo's canonical LEGACY_PTM_TO_UNIMOD table: the
    modification's legacy token is reconstructed as residue + delta (e.g.
    "M(+15.99)", plus N-terminal forms such as "(+42.01)") and looked up there.
    A unimod_id already set by the extractor is trusted directly. As a last
    resort -- used when the table is unavailable (no InstaNovo package) or lacks
    the token -- the delta mass is matched against the tracked PTMs. Returns -1
    if unresolved.
    """
    uid = int(mod.get("unimod_id", -1))
    if uid > 0:
        return uid

    pos = int(mod.get("position", 0))
    mod_name = str(mod.get("mod_name", ""))
    tokens: list[str] = []
    if 1 <= pos <= len(peptide):
        tokens.append(f"{peptide[pos - 1]}({mod_name})")
    else:
        # N-terminal modifications are commonly keyed without a residue, but
        # some residue sets encode N-terminal chemistry on the first residue.
        tokens.append(f"({mod_name})")
        if peptide:
            tokens.append(f"{peptide[0]}({mod_name})")

    for token in tokens:
        match = _UNIMOD_ID_RE.search(LEGACY_PTM_TO_UNIMOD.get(token, ""))
        if match:
            return int(match.group(1))

    delta = _mod_delta_mass(mod)
    if delta is not None:
        for unimod, (_name, mass) in TRACKED_PTMS.items():
            if abs(abs(delta) - mass) <= PTM_MASS_TOL:
                return unimod
    return -1


def merge_spectrum_concepts(
    token_concepts: dict[str, bool],
    spectrum_concepts: dict[str, bool],
) -> dict[str, bool]:
    """Apply spectrum-level labels to one token without clobbering token labels.

    Spectrum-level concepts are intentionally replicated to every token in a
    spectrum, including noise peaks. Use OR semantics so a future concept name
    collision cannot turn off a peak-level label that was already true.
    """
    for cname, val in spectrum_concepts.items():
        token_concepts[cname] = bool(token_concepts.get(cname, False)) or bool(val)
    return token_concepts


def _mod_to_ptm_name(mod: dict, peptide: str) -> str | None:
    """Map a modification to a tracked PTM concept name, or None if untracked."""
    entry = TRACKED_PTMS.get(_mod_unimod_id(mod, peptide))
    return entry[0] if entry else None


def exact_mass_proforma(proforma: str) -> str:
    """Substitute the dataset's rounded delta masses with exact PTM masses.

    The benchmark rounds modification deltas to two decimals; at a tight (ppm)
    fragment tolerance that rounding can drop a modified fragment out of the match
    window. UNIMOD ProForma is not an option here because spectrum_utils resolves
    UNIMOD ids over the network, so each rounded delta that matches a tracked PTM
    by mass is replaced with the exact monoisotopic mass instead. UNIMOD tags and
    unrecognised deltas are left untouched.
    """
    def repl(m: "re.Match") -> str:
        body = m.group(1)
        if "UNIMOD" in body.upper():
            return m.group(0)
        try:
            val = float(body)
        except ValueError:
            return m.group(0)
        for _name, mass in TRACKED_PTMS.values():
            if abs(abs(val) - mass) <= PTM_MASS_TOL:
                sign = "+" if val >= 0 else "-"
                return f"[{sign}{mass:.6f}]"
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]", repl, proforma)


# ─────────────────────────────────────────────────────────────────────────────
# spectrum_utils adapter (spectrum_utils 0.5.x)
# ─────────────────────────────────────────────────────────────────────────────
# In 0.5.x each peak carries a PeakInterpretation with a list of FragmentAnnotation
# objects. Each FragmentAnnotation has a combined ion_type string ("b2", "y3",
# "IC" for the Cys immonium, "m3:5" for an internal fragment, "p" for precursor),
# plus charge, neutral_loss (e.g. "-H2O"), isotope, and mz_delta. This is the one
# place that depends on that layout.
_BACKBONE_RE = re.compile(r"^([abcxyz])(\d+)$")
_INTERNAL_RE = re.compile(r"^m(\d+):(\d+)$")
_IMMONIUM_RE = re.compile(r"^I([A-Z])")

# Preference order when a peak has several candidate annotations (lower = better).
_ION_PRIORITY = {"b": 0, "y": 0, "a": 1, "c": 1, "x": 1, "z": 1, "I": 2, "internal": 3, "precursor": 4}

_NOISE = {"ion_type": "noise", "ion_index": None, "charge": 1, "neutral_loss": ""}


def _parse_ion_type(ion_type_str: str) -> tuple:
    """Parse a spectrum_utils ion_type string into (type, index).

    index is an int for backbone ions, a (start, end) tuple for internal
    fragments, the residue letter for immonium ions, and None for the precursor.
    Returns (None, None) for anything unrecognised.
    """
    s = str(ion_type_str)
    m = _BACKBONE_RE.match(s)
    if m:
        return m.group(1), int(m.group(2))
    m = _IMMONIUM_RE.match(s)
    if m:
        return "I", m.group(1)
    m = _INTERNAL_RE.match(s)
    if m:
        return "internal", (int(m.group(1)), int(m.group(2)))
    if s == "p":
        return "precursor", None
    return None, None


def _annotation_to_dict(peak_interpretation) -> dict:
    """Reduce one spectrum_utils PeakInterpretation to a flat label dict.

    Picks the best candidate annotation (monoisotopic first, then no neutral loss,
    then canonical b/y over other ions, then lowest charge and smallest mass
    error) and maps it to {ion_type, ion_index, charge, neutral_loss}. Peaks with
    no annotation become noise.
    """
    frags = getattr(peak_interpretation, "fragment_annotations", None) or []
    candidates = []
    for fa in frags:
        ion_type, ion_index = _parse_ion_type(getattr(fa, "ion_type", ""))
        if ion_type is None:
            continue
        nl_raw = getattr(fa, "neutral_loss", None)
        neutral_loss = str(nl_raw).lstrip("-") if nl_raw else ""
        charge = int(getattr(fa, "charge", 1))
        isotope = int(getattr(fa, "isotope", 0) or 0)
        mz_delta = getattr(fa, "mz_delta", None)
        delta_abs = abs(mz_delta[0]) if mz_delta else 0.0
        sort_key = (
            isotope != 0,
            bool(neutral_loss),
            _ION_PRIORITY.get(ion_type, 5),
            charge,
            delta_abs,
        )
        candidates.append((sort_key, {
            "ion_type": ion_type,
            "ion_index": ion_index,
            "charge": charge,
            "neutral_loss": neutral_loss,
        }))

    if not candidates:
        return dict(_NOISE)
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def run_spectrum_utils(
    proforma: str,
    mz_array: torch.Tensor,
    intensity_array: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
    ion_types: str,
    fragment_tol_mass: float,
    fragment_tol_mode: str,
    enable_internal: bool,
) -> list[dict]:
    """Annotate one spectrum and return a flat label dict per peak."""
    mz_np = mz_array.detach().cpu().numpy().astype(np.float64)
    int_np = intensity_array.detach().cpu().numpy().astype(np.float64)

    spectrum = sus.MsmsSpectrum(
        identifier="",
        precursor_mz=precursor_mz,
        precursor_charge=max(1, precursor_charge),
        mz=mz_np,
        intensity=int_np,
    )

    # Internal fragments are ion type 'm' in spectrum_utils, not 'by'.
    annotation_ion_types = ion_types
    if enable_internal and "m" not in annotation_ion_types:
        annotation_ion_types += "m"

    try:
        spectrum.annotate_proforma(
            exact_mass_proforma(proforma),               # exact PTM masses; positional arg
            fragment_tol_mass,
            fragment_tol_mode,
            ion_types=annotation_ion_types,
            max_isotope=0,                               # monoisotopic fragments only
            neutral_losses={"H2O": -18.010565, "NH3": -17.026549},
        )
    except Exception as exc:
        # Bad ProForma or unsupported residue — return all-noise to keep the run going.
        LOG.warning("spectrum_utils annotation failed for %r: %s", proforma, exc)
        return [dict(_NOISE) for _ in range(len(mz_np))]

    annotations = getattr(spectrum, "annotation", None)
    if annotations is None:
        return [dict(_NOISE) for _ in range(len(mz_np))]
    return [_annotation_to_dict(a) for a in annotations]


# ─────────────────────────────────────────────────────────────────────────────
# Concept computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_spectrum_concepts(
    modifications: list[dict],
    precursor_charge: int,
    peptide: str,
) -> dict[str, bool]:
    """Concepts that depend only on spectrum-level properties (not on the
    current peak). These are replicated across all tokens of a spectrum.
    """
    out: dict[str, bool] = {}
    peptide_length = len(peptide)

    # Precursor charge buckets.
    for c in (2, 3, 4):
        out[f"precursor_charge_{c}"] = (precursor_charge == c)

    # Peptide length buckets.
    short_cut, long_cut = PEPTIDE_LEN_CUTS
    out["peptide_length_short"] = peptide_length < short_cut
    out["peptide_length_medium"] = short_cut <= peptide_length < long_cut
    out["peptide_length_long"] = peptide_length >= long_cut

    # Spectrum-level PTM diagnostics: does any modification map to each tracked PTM.
    present = {_mod_to_ptm_name(m, peptide) for m in modifications}
    for _unimod, (ptm_name, _mass) in TRACKED_PTMS.items():
        out[f"spectrum_contains_{ptm_name}"] = ptm_name in present

    return out


def compute_top_decile_threshold(intensities: torch.Tensor) -> float:
    """Per-spectrum intensity threshold for the top-decile concept.

    Computed per-spectrum so the concept's prevalence stays calibrated
    regardless of spectrum-to-spectrum dynamic range differences.
    """
    if intensities.numel() == 0:
        return float("inf")
    # 90th percentile: top 10% of peaks fire above this.
    return float(torch.quantile(intensities.float(), 0.9))


def compute_peak_concepts(
    annotation: dict,
    mz: float,
    intensity: float,
    intensity_threshold: float,
    peptide: str,
    modifications: list[dict],
    peptide_length: int,
    registry: ConceptRegistry,
) -> dict[str, bool]:
    """Concepts that depend on the specific peak being labeled."""
    out: dict[str, bool] = {n: False for n in registry.names}

    ion_type = annotation["ion_type"]
    ion_index = annotation["ion_index"]
    fragment_charge = annotation["charge"]
    neutral_loss = annotation["neutral_loss"]

    # ─── Ion-type family ───
    out["is_b_ion"] = ion_type == "b"
    out["is_y_ion"] = ion_type == "y"
    out["is_I_ion"] = ion_type == "I"
    out["is_internal_fragment"] = ion_type == "internal"
    out["is_precursor_related"] = ion_type == "precursor"
    out["is_noise_peak"] = ion_type == "noise"
    out["is_latent_token"] = False  # latent handled separately
    out["is_matched_peak"] = ion_type not in ("noise", "latent")

    # ─── Neutral loss family ───
    nl = neutral_loss or ""
    out["is_H2O_loss"] = "H2O" in nl
    out["is_NH3_loss"] = "NH3" in nl
    out["has_neutral_loss"] = bool(nl)

    # ─── Fragment charge family ───
    out["is_fragment_charge_1"] = fragment_charge == 1
    out["is_fragment_charge_2"] = fragment_charge == 2

    # ─── Mass region family ───
    low_cut, high_cut = MZ_LOW_HIGH_CUT
    out["mz_low"] = mz < low_cut
    out["mz_mid"] = low_cut <= mz < high_cut
    out["mz_high"] = mz >= high_cut

    # ─── Intensity family ───
    out["top_decile_intensity"] = intensity >= intensity_threshold

    # The remaining concepts require ion geometry.
    geom = IonGeometry(ion_type, ion_index, peptide_length)

    # ─── Position family (for matched b/y) ───
    # b_k / y_k: k is the cleavage index. Map to N/middle/C terms based on
    # where the cleavage falls along the peptide.
    if ion_type in ("b", "y") and isinstance(geom.ion_index, int) and geom.ion_index > 0:
        cleavage_pos = geom.ion_index if ion_type == "b" else (peptide_length - geom.ion_index)
        third = peptide_length / 3.0
        out["position_Nterm"] = cleavage_pos < third
        out["position_middle"] = third <= cleavage_pos < 2 * third
        out["position_Cterm"] = cleavage_pos >= 2 * third

    # ─── First/last ion family ───
    if ion_type in ("b", "y") and isinstance(geom.ion_index, int):
        out["is_first_ion"] = geom.ion_index == 1
        out["is_last_ion"] = geom.ion_index == peptide_length - 1

    # ─── Cleavage chemistry families ───
    # A b_k or y_(L-k) ion implies a cleavage between residues k and k+1 (1-indexed).
    cleavage_after_idx = None
    if ion_type == "b" and isinstance(geom.ion_index, int) and 1 <= geom.ion_index < peptide_length:
        cleavage_after_idx = geom.ion_index
    elif ion_type == "y" and isinstance(geom.ion_index, int) and 1 <= geom.ion_index < peptide_length:
        cleavage_after_idx = peptide_length - geom.ion_index

    if cleavage_after_idx is not None and 1 <= cleavage_after_idx < peptide_length:
        before = peptide[cleavage_after_idx - 1]
        after = peptide[cleavage_after_idx]
        out["cleaves_C_to_Lys"] = (before == "K")
        out["cleaves_C_to_Arg"] = (before == "R")
        out["cleaves_C_to_Asp"] = (before == "D")
        out["cleaves_C_to_Glu"] = (before == "E")
        out["cleaves_N_to_Pro"] = (after == "P")

    # ─── Residue-cover family ───
    for residue in "KRWFYPMCDEN":
        concept_name = f"covers_{residue}"
        if concept_name in registry.names:
            out[concept_name] = geom.covers_any_residue(peptide, residue)

    # ─── Token-level PTM containment ───
    # A b/y/internal ion that spans the modified residue's position contains it;
    # immonium ions don't localise (geom.covers returns False for them).
    for mod in modifications:
        mod_pos = mod.get("position", 0)
        if mod_pos < 1 or mod_pos > peptide_length:
            continue
        ptm_name = _mod_to_ptm_name(mod, peptide)
        if ptm_name is None:
            continue
        concept_name = f"ion_contains_{ptm_name}"
        if concept_name in registry.names and geom.covers(mod_pos):
            out[concept_name] = True

    return out


def compute_latent_concepts(
    spectrum_concepts: dict[str, bool],
    peptide: str,
    modifications: list[dict],
    peptide_length: int,
    registry: ConceptRegistry,
) -> dict[str, bool]:
    """Concept dict for the latent token at position 0.

    The latent covers the whole peptide, so residue-cover and token-PTM
    concepts apply if the residue / modification exists anywhere. All
    peak-specific concepts (m/z bins, ion type, neutral loss) are False.
    """
    out: dict[str, bool] = {n: False for n in registry.names}
    merge_spectrum_concepts(out, spectrum_concepts)
    out["is_latent_token"] = True

    # Latent covers whole peptide for residue-cover concepts.
    for residue in "KRWFYPMCDEN":
        concept_name = f"covers_{residue}"
        if concept_name in registry.names:
            out[concept_name] = residue in peptide

    # Latent contains all PTMs the spectrum contains.
    for mod in modifications:
        ptm_name = _mod_to_ptm_name(mod, peptide)
        if ptm_name is None:
            continue
        concept_name = f"ion_contains_{ptm_name}"
        if concept_name in registry.names:
            out[concept_name] = True

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk annotation
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class AnnotationConfig:
    """Configuration for an annotation run."""

    extract_dir: Path
    output_dir: Path
    ion_types: str = "byIp"             # b, y, immonium, precursor (internal added via flag)
    enable_internal: bool = True        # annotate internal fragments (ion type 'm')
    fragment_tol_mass: float = 20.0     # fragment match tolerance (see fragment_tol_mode)
    fragment_tol_mode: str = "ppm"      # 'ppm' (high-res Orbitrap) or 'Da'
    resume: bool = True

    def as_jsonable(self) -> dict:
        out = dataclasses.asdict(self)
        out["extract_dir"] = str(self.extract_dir)
        out["output_dir"] = str(self.output_dir)
        return out


def annotate_chunk(
    chunk_data: dict,
    config: AnnotationConfig,
    registry: ConceptRegistry,
) -> LabelChunkData:
    """Build a LabelChunkData for one input chunk.

    `chunk_data` is a v4 ChunkMeta loaded as a dict. The output token ordering
    matches the chunk's flattening — for each spectrum, the latent token at
    position 0 followed by one token per peak — so the result row-aligns with
    activations[layer] for that chunk.
    """
    n_spectra = chunk_data["n_spectra"]
    chunk_idx = chunk_data["chunk_idx"]
    name_to_idx = registry.index
    n_concepts = len(registry.names)

    # Accumulate per-spectrum tensors and concatenate at the end. Cheaper
    # than indexing into a pre-allocated tensor token-by-token in Python.
    label_parts: list[torch.Tensor] = []
    ion_type_parts: list[torch.Tensor] = []
    ion_index_parts: list[torch.Tensor] = []
    fragment_charge_parts: list[torch.Tensor] = []
    nl_id_parts: list[torch.Tensor] = []
    mz_parts: list[torch.Tensor] = []
    intensity_parts: list[torch.Tensor] = []

    for s in range(n_spectra):
        peptide = chunk_data["peptides"][s]
        proforma = chunk_data["proforma_strings"][s]
        modifications = chunk_data["modifications"][s]
        mz_array = chunk_data["mz_arrays"][s]
        intensity_array = chunk_data["intensity_arrays"][s]
        precursor_charge = int(chunk_data["precursor_charges"][s])
        precursor_mz = float(chunk_data["precursor_mzs"][s])
        peptide_length = len(peptide)
        n_peaks = len(mz_array)
        n_tokens_s = 1 + n_peaks  # latent + peaks

        # Pre-compute spectrum-level concepts and intensity threshold once.
        spectrum_concepts = compute_spectrum_concepts(
            modifications, precursor_charge, peptide,
        )
        intensity_threshold = compute_top_decile_threshold(intensity_array)

        # Annotate the spectrum; returns one label dict per peak (noise if unmatched).
        annotations = run_spectrum_utils(
            proforma=proforma,
            mz_array=mz_array,
            intensity_array=intensity_array,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            ion_types=config.ion_types,
            fragment_tol_mass=config.fragment_tol_mass,
            fragment_tol_mode=config.fragment_tol_mode,
            enable_internal=config.enable_internal,
        )

        # Allocate per-spectrum tensors and fill row by row.
        labels_s = torch.zeros(n_tokens_s, n_concepts, dtype=torch.bool)
        ion_type_s = torch.zeros(n_tokens_s, dtype=torch.int8)
        ion_index_s = torch.full((n_tokens_s,), -1, dtype=torch.int16)
        fragment_charge_s = torch.zeros(n_tokens_s, dtype=torch.int8)
        nl_id_s = torch.zeros(n_tokens_s, dtype=torch.int8)
        mz_s = torch.zeros(n_tokens_s, dtype=torch.float32)
        intensity_s = torch.zeros(n_tokens_s, dtype=torch.float32)

        # Position 0: latent token.
        latent_concepts = compute_latent_concepts(
            spectrum_concepts, peptide, modifications, peptide_length, registry,
        )
        for cname, val in latent_concepts.items():
            if val:
                labels_s[0, name_to_idx[cname]] = True
        ion_type_s[0] = ION_TYPE_VOCAB["latent"]

        # Positions 1..n_peaks: peak tokens.
        for peak_idx, ann in enumerate(annotations):
            pos = peak_idx + 1
            mz = float(mz_array[peak_idx])
            intensity = float(intensity_array[peak_idx])

            peak_concepts = compute_peak_concepts(
                ann, mz, intensity, intensity_threshold,
                peptide, modifications, peptide_length,
                registry,
            )
            # Spectrum-level concepts apply to every peak too. Merge with OR
            # semantics so spectrum labels cannot clobber peak-local labels if a
            # future concept family reuses a name.
            merge_spectrum_concepts(peak_concepts, spectrum_concepts)

            for cname, val in peak_concepts.items():
                if val:
                    labels_s[pos, name_to_idx[cname]] = True

            ion_type_s[pos] = ION_TYPE_VOCAB.get(ann["ion_type"], 0)
            ii = ann["ion_index"]
            if isinstance(ii, int) and ii >= 0:
                ion_index_s[pos] = min(ii, 32_000)
            elif isinstance(ii, tuple):
                # Internal fragment: store the start position only.
                ion_index_s[pos] = min(ii[0], 32_000)
            fragment_charge_s[pos] = min(ann["charge"], 127)
            nl_id_s[pos] = NEUTRAL_LOSS_VOCAB.get(ann["neutral_loss"] or "", 0)
            mz_s[pos] = mz
            intensity_s[pos] = intensity

        label_parts.append(labels_s)
        ion_type_parts.append(ion_type_s)
        ion_index_parts.append(ion_index_s)
        fragment_charge_parts.append(fragment_charge_s)
        nl_id_parts.append(nl_id_s)
        mz_parts.append(mz_s)
        intensity_parts.append(intensity_s)

    token_labels = torch.cat(label_parts, dim=0)
    return LabelChunkData(
        schema_version=SCHEMA_VERSION,
        chunk_idx=chunk_idx,
        n_spectra=n_spectra,
        total_tokens=int(token_labels.size(0)),
        concept_names=list(registry.names),
        token_labels=token_labels,
        ion_type_ids=torch.cat(ion_type_parts, dim=0),
        ion_indices=torch.cat(ion_index_parts, dim=0),
        fragment_charges=torch.cat(fragment_charge_parts, dim=0),
        neutral_loss_ids=torch.cat(nl_id_parts, dim=0),
        peak_mzs=torch.cat(mz_parts, dim=0),
        peak_intensities=torch.cat(intensity_parts, dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming aggregation of base rates and concept-concept correlations
# ─────────────────────────────────────────────────────────────────────────────
class ConceptStatsAccumulator:
    """Accumulate co-occurrence counts across chunks for the manifest.

    The phi-coefficient correlation matrix is derived from these counts
    at the end of the run.
    """

    def __init__(self, n_concepts: int):
        self.n_concepts = n_concepts
        self.cooccur = torch.zeros((n_concepts, n_concepts), dtype=torch.float64)
        self.marginal = torch.zeros(n_concepts, dtype=torch.float64)
        self.total = 0

    def add_chunk(self, labels_bool: torch.Tensor) -> None:
        """Update accumulators with one chunk's [n_tokens, n_concepts] labels."""
        x = labels_bool.to(torch.float64)
        self.cooccur += x.T @ x
        self.marginal += x.sum(dim=0)
        self.total += x.size(0)

    def base_rates(self) -> torch.Tensor:
        return self.marginal / max(self.total, 1)

    def phi_matrix(self) -> torch.Tensor:
        """Pairwise phi coefficient (Pearson correlation for binary variables)."""
        n11 = self.cooccur
        ni = self.marginal.unsqueeze(1)        # [C, 1]
        nj = self.marginal.unsqueeze(0)        # [1, C]
        N = float(self.total)

        n10 = ni - n11
        n01 = nj - n11
        n00 = N - ni - nj + n11

        numerator = n11 * n00 - n10 * n01
        denom = torch.sqrt((n11 + n10) * (n00 + n01) * (n11 + n01) * (n00 + n10))
        denom = denom.clamp_min(1e-12)
        phi = numerator / denom
        # Diagonal is 1 by definition; undefined pairs sit at 0 from the clamp.
        phi.fill_diagonal_(1.0)
        return phi.to(torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
class AnnotationRunner:
    """Walks the extraction manifest, annotates each chunk, writes labels,
    and emits the annotation_manifest.json with global statistics.
    """

    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.registry = ConceptRegistry.build()
        self.labels_dir = config.output_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = config.output_dir / "annotation_manifest.json"

        extract_manifest_path = config.extract_dir / "manifest.json"
        if not extract_manifest_path.exists():
            raise FileNotFoundError(f"Extract manifest not found at {extract_manifest_path}")
        self.extract_manifest = json.loads(extract_manifest_path.read_text())

        if self.extract_manifest["schema_version"] != EXTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"Extraction schema mismatch: manifest={self.extract_manifest['schema_version']}, "
                f"annotator expects {EXTRACT_SCHEMA_VERSION}. Re-run extract.py."
            )

        LOG.info("Annotation registry: %d concepts across %d families",
                 len(self.registry.names), len(CONCEPT_FAMILIES))

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self.labels_dir / f"chunk_{chunk_idx:05d}.pt"

    def _load_extract_chunk(self, chunk_idx: int) -> dict:
        # v4: per-chunk metadata lives in its own meta file; the annotator never
        # touches the activation tensors, so only the ChunkMeta is loaded.
        meta_rel = self.extract_manifest["chunks"][chunk_idx]["meta"]
        path = self.config.extract_dir / meta_rel
        return torch.load(path, map_location="cpu", weights_only=False)

    def run(self) -> None:
        accumulator = ConceptStatsAccumulator(len(self.registry.names))
        n_chunks = self.extract_manifest["n_chunks"]
        t0 = time.time()

        for chunk_idx in range(n_chunks):
            out_path = self._chunk_path(chunk_idx)

            if self.config.resume and out_path.exists():
                # Re-load existing labels so accumulator picks them up for stats.
                LOG.info("Chunk %d already annotated, loading for stats", chunk_idx)
                existing = LabelChunkData.load(out_path)
                accumulator.add_chunk(existing.token_labels)
                continue

            LOG.info("Annotating chunk %d / %d", chunk_idx + 1, n_chunks)
            chunk = self._load_extract_chunk(chunk_idx)
            label_chunk = annotate_chunk(chunk, self.config, self.registry)
            label_chunk.save(out_path)
            accumulator.add_chunk(label_chunk.token_labels)
            LOG.info(
                "  Saved %s | n_tokens=%d | elapsed=%.1fs",
                out_path.name, label_chunk.total_tokens, time.time() - t0,
            )

        self._write_manifest(accumulator, n_chunks)
        LOG.info(
            "Annotation complete: %d chunks, %d tokens, %.0fs",
            n_chunks, accumulator.total, time.time() - t0,
        )

    def _write_manifest(self, accumulator: ConceptStatsAccumulator, n_chunks: int) -> None:
        base_rates = accumulator.base_rates().tolist()
        phi = accumulator.phi_matrix()

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_jsonable(),
            "registry": self.registry.to_jsonable(),
            "n_chunks": n_chunks,
            "n_tokens": int(accumulator.total),
            "base_rates": {
                name: float(base_rates[i])
                for i, name in enumerate(self.registry.names)
            },
            "vocab": {
                "ion_type": ION_TYPE_VOCAB,
                "neutral_loss": NEUTRAL_LOSS_VOCAB,
            },
            "chunks": [
                {"idx": i, "path": str(self._chunk_path(i).relative_to(self.config.output_dir))}
                for i in range(n_chunks)
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

        # Phi matrix is large; save as torch tensor alongside the JSON manifest.
        phi_path = self.config.output_dir / "concept_phi.pt"
        torch.save({
            "concept_names": self.registry.names,
            "phi": phi,
            "marginal": accumulator.marginal.to(torch.float32),
            "cooccur": accumulator.cooccur.to(torch.float32),
            "n_tokens": int(accumulator.total),
        }, phi_path)
        LOG.info("Wrote manifest to %s and phi matrix to %s", self.manifest_path, phi_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--extract-dir", type=Path, required=True,
                   help="Directory containing manifest.json from extract.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ion-types", default="byIp",
                   help="spectrum_utils ion_types string (b, y, immonium I, precursor p). "
                        "Internal fragments (ion type 'm') are added when --enable-internal.")
    p.add_argument("--no-internal", action="store_true",
                   help="Disable internal fragment annotation.")
    p.add_argument("--fragment-tol", type=float, default=20.0,
                   help="Fragment mass tolerance (units set by --fragment-tol-mode).")
    p.add_argument("--fragment-tol-mode", default="ppm", choices=["ppm", "Da"],
                   help="Fragment tolerance unit; ppm for high-res Orbitrap data.")
    p.add_argument("--no-resume", action="store_true",
                   help="Force re-annotation of chunks that already exist.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    config = AnnotationConfig(
        extract_dir=args.extract_dir,
        output_dir=args.output_dir,
        ion_types=args.ion_types,
        enable_internal=not args.no_internal,
        fragment_tol_mass=args.fragment_tol,
        fragment_tol_mode=args.fragment_tol_mode,
        resume=not args.no_resume,
    )

    runner = AnnotationRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
