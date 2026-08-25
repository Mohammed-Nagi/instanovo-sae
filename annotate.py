"""Per-token concept annotation for InstaNovo SAE evaluation.

Reads the per-chunk metadata written by extract.py (ChunkMeta only -- activations
are never needed here) and writes one LabelChunkData per chunk in the same token
order, so labels join row-for-row against activations with no index translation.
That alignment is the module's contract and is checked in annotate_chunk.

Annotation
    spectrum_utils 0.5.x supplies the mzPAF-style fragment annotations, and
    _annotation_to_dict is the only place that reads its object layout, so a
    version bump stays localised there.

    Matching uses a ppm tolerance (20 ppm default) rather than a fixed Da window:
    at Orbitrap resolution a 0.5 Da window matches many spurious peaks.

    Modification deltas are replaced with exact monoisotopic masses first. The
    dataset rounds them to two decimals ("+15.99" for oxidation, exactly
    15.994915), and at 20 ppm that rounding can push a modified fragment out of
    tolerance.

    Only monoisotopic fragments are generated (max_isotope=0), so isotope peaks
    of a real fragment fall through to the noise label -- the conservative choice
    for per-token labelling.

PTM identity
    Comes from InstaNovo's LEGACY_PTM_TO_UNIMOD table: the same mapping the model
    uses to build its residue vocabulary, so labels describe what the encoder
    saw. Fragment annotation still passes exact delta masses rather than UNIMOD
    tags, because spectrum_utils resolves UNIMOD ids over the network. Tracked
    PTMs are oxidation, carbamidomethyl-Cys and deamidation; mass matching is a
    fallback for when the table is unavailable.

Concepts
    50 concepts in 14 families: ion type, neutral loss, fragment charge,
    precursor charge, position, mass region, intensity, first/last ion, canonical
    and enhanced cleavage, residue cover, peptide length, and token- versus
    spectrum-level PTM containment (the spectrum-level set is a negative control).

    Immonium ions attest that a residue is PRESENT, not where it sits, so they
    drive only is_I_ion and the matching covers_<residue> -- never position,
    cleavage, or PTM-localisation concepts.

Output layout under --output-dir
    annotation_manifest.json   registry, base rates, vocabularies
    concept_phi.pt             concept-concept correlation matrix + counts
    labels/
        chunk_00000.pt         one LabelChunkData per input chunk
        ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# Fragment-ion annotation engine; its object layout is read only in
# _annotation_to_dict (see module docstring).
import spectrum_utils.spectrum as sus

from schema import ANNOTATION_SCHEMA_VERSION, EXTRACT_SCHEMA_VERSION

# Canonical legacy-token -> UNIMOD table. Guarded so this module still runs
# without the InstaNovo package; PTM identity then falls back to mass matching
# (see _mod_unimod_id).
try:
    from instanovo.constants import LEGACY_PTM_TO_UNIMOD
except Exception:
    LEGACY_PTM_TO_UNIMOD: dict[str, str] = {}

LOG = logging.getLogger("annotate")


def atomic_torch_save(obj, path: Path) -> None:
    """torch.save to a temp file in the same directory, then rename into place.

    Resume here is an existence check (`out_path.exists()`), so a file that
    exists is assumed complete. A process killed mid-write would otherwise leave
    a truncated label chunk that the next run silently accepts -- and with
    --num-workers > 1 a single Ctrl-C now interrupts several concurrent writes
    at once. os.replace is atomic within a filesystem.

    Duplicated from extract.py rather than imported: annotate.py deliberately
    does not depend on the model-dependent extraction module.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _init_annotation_worker() -> None:
    """Per-worker setup for the ProcessPoolExecutor path.

    Each worker inherits torch's default intra-op thread count, which is the
    machine's core count -- so N workers on an N-core box request N*N threads
    and spend most of their time contending rather than annotating. Annotation
    parallelism is across chunks, so one thread per worker is what we want.
    """
    torch.set_num_threads(1)


# --- Concept registry ---------------------------------------------------------
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
# ablation -- they exist as negative controls for methodology validation.
DIAGNOSTIC_FAMILIES: set[str] = {"spectrum_ptm_diagnostic"}

# Residues with a covers_<residue> concept. Derived from the registry so the
# loop and the concept list cannot drift apart.
COVERED_RESIDUES: str = "".join(
    c.removeprefix("covers_") for c in CONCEPT_FAMILIES["residue_cover"]
)

# Mass-region bin boundaries in Da.
MZ_LOW_HIGH_CUT = (300.0, 1000.0)

# Peptide-length bin boundaries (inclusive on left).
PEPTIDE_LEN_CUTS = (7, 16)  # short < 7, medium 7..15, long >= 16

# UNIMOD id -> (concept short name, exact monoisotopic mass in Da). The only
# modifications with non-trivial base rate in the benchmark. Masses drive
# exact_mass_proforma (fragment matching); names drive the ion_contains_* and
# spectrum_contains_* concepts; identity comes from _mod_unimod_id.
TRACKED_PTMS: dict[int, tuple[str, float]] = {
    35: ("oxidation",   15.994915),   # "+15.99" on M
    4:  ("cam_cys",     57.021464),   # "+57.02" on C (carbamidomethyl)
    7:  ("deamidation",  0.984016),   # "+0.98"/"+.98" on N/Q
}
# Tolerance for the mass-matching fallback. The dataset rounds to two decimals
# (~0.005 Da error) and the tracked masses are far apart, so this is unambiguous.
PTM_MASS_TOL = 0.02

# Neutral losses passed to spectrum_utils, as exact monoisotopic deltas.
NEUTRAL_LOSSES = {"H2O": -18.010565, "NH3": -17.026549}

_MASS_DELTA_RE = re.compile(r"[+-]?\d*\.?\d+")
_UNIMOD_ID_RE = re.compile(r"UNIMOD:(\d+)")
_PROFORMA_TAG_RE = re.compile(r"\[([^\]]+)\]")

# Small int vocabularies for compact per-token metadata storage.
ION_TYPE_VOCAB: dict[str, int] = {
    "noise": 0, "latent": 1, "precursor": 2,
    "b": 3, "y": 4, "I": 5, "internal": 6,
    "a": 7, "c": 8, "x": 9, "z": 10,
}

NEUTRAL_LOSS_VOCAB: dict[str, int] = {
    "": 0, "H2O": 1, "NH3": 2, "H2O+NH3": 3, "H3PO4": 4,
}

# Storage caps for the compact per-token metadata tensors (int16 / int8).
_ION_INDEX_CAP = 32_000
_FRAGMENT_CHARGE_CAP = 127


@dataclasses.dataclass
class ConceptRegistry:
    """The full set of concepts, with family memberships and a name->index map."""

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
        registry = cls(names=names, family_of=family_of, diagnostic=diagnostic)
        registry._check_derived_names()
        return registry

    def _check_derived_names(self) -> None:
        """Fail at start-up if a concept the annotator writes is not registered.

        The per-token paths build concept names from COVERED_RESIDUES and
        TRACKED_PTMS, constants separate from CONCEPT_FAMILIES. Checking that
        coupling once here lets those paths assign directly rather than testing
        registry membership for every residue and PTM on every token.
        """
        expected = (
            [f"covers_{r}" for r in COVERED_RESIDUES]
            + [f"ion_contains_{n}" for n in self.ptm_keys]
            + [f"spectrum_contains_{n}" for n in self.ptm_keys]
        )
        registered = set(self.names)
        missing = [n for n in expected if n not in registered]
        if missing:
            raise ValueError(
                f"Concept registry is missing names the annotator writes: {missing}. "
                "CONCEPT_FAMILIES must cover COVERED_RESIDUES and TRACKED_PTMS."
            )

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


@dataclasses.dataclass
class LabelChunkData:
    """Per-token labels for one input chunk, in the same token order and count as
    the corresponding ChunkMeta, so the evaluator joins them row-for-row against
    activations[layer] without index translation.
    """

    schema_version: int
    chunk_idx: int
    n_spectra: int
    total_tokens: int

    # Concept names for self-description; full registry lives in the manifest.
    concept_names: list[str]

    # Main payload: per-token boolean labels [total_tokens, n_concepts].
    # torch.bool is ~6x smaller than float; consumers cast as needed.
    token_labels: torch.Tensor

    # Per-token metadata for analysis tools (filter without re-derivation).
    ion_type_ids: torch.Tensor       # [total_tokens] int8, decode via ION_TYPE_VOCAB
    ion_indices: torch.Tensor        # [total_tokens] int16, -1 if N/A
    fragment_charges: torch.Tensor   # [total_tokens] int8
    neutral_loss_ids: torch.Tensor   # [total_tokens] int8, decode via NEUTRAL_LOSS_VOCAB
    peak_mzs: torch.Tensor           # [total_tokens] float32, 0 for latent
    peak_intensities: torch.Tensor   # [total_tokens] float32, 0 for latent

    def save(self, path: Path) -> None:
        # Shallow field dict rather than dataclasses.asdict: asdict deep-copies
        # every value, duplicating the label tensors in RAM to write a
        # byte-identical file.
        atomic_torch_save(dict(self.__dict__), path)

    @classmethod
    def load(cls, path: Path) -> LabelChunkData:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


class IonGeometry:
    """Which 1-indexed residues of the parent peptide a fragment ion spans.

    This is what makes token-level PTM and residue-cover labels possible.
    Immonium ions are the exception: they attest a residue's presence anywhere
    in the peptide rather than at a position, so they cover their residue type
    only and can never localise a PTM.
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
            # attests that residue type, but not where.
            return residue == self.residue
        for i, aa in enumerate(peptide, start=1):
            if aa == residue and self.covers(i):
                return True
        return False

    def backbone_index(self) -> int | None:
        """The backbone cleavage index k for a b/y ion, else None.

        Several concept families are defined only for backbone ions with an
        integer index; this centralises that guard.
        """
        if self.ion_type in ("b", "y") and isinstance(self.ion_index, int):
            return self.ion_index
        return None


# --- Modification helpers -----------------------------------------------------
# UNIMOD identity via the canonical table, plus exact-mass ProForma rewriting.

def _mod_delta_mass(mod: dict) -> float | None:
    """Numeric delta mass from a modification's mod_name; None for a UNIMOD tag
    (handled separately) or an unparseable name."""
    name = str(mod.get("mod_name", ""))
    if "UNIMOD" in name.upper():
        return None
    m = _MASS_DELTA_RE.search(name)
    return float(m.group(0)) if m else None


# _to_proforma (extract.py) zero-pads bare-dot deltas for ProForma parsing, e.g.
# the dataset's deamidation "(+.98)" becomes "[+0.98]", so mod_name arrives here
# already zero-padded. InstaNovo's LEGACY_PTM_TO_UNIMOD is keyed on the dataset's
# raw, un-padded form ("N(+.98)"), so the padded string alone never matches --
# the canonical lookup would silently miss every such PTM and fall through to
# the mass-tolerance fallback. Stripping the pad back out lets both forms match.
_ZERO_PADDED_DELTA_RE = re.compile(r"^([+-])0(\.\d+)$")


def _legacy_ptm_tokens(mod_name: str, position: int, peptide: str) -> list[str]:
    """Candidate LEGACY_PTM_TO_UNIMOD keys for a modification.

    In-sequence modifications are keyed as residue + delta ("M(+15.99)").
    N-terminal ones are usually keyed without a residue, though some residue sets
    put the chemistry on the first residue, so both forms are tried -- each in
    both delta spellings (see _ZERO_PADDED_DELTA_RE).
    """
    names = [mod_name]
    unpadded = _ZERO_PADDED_DELTA_RE.match(mod_name)
    if unpadded:
        names.append(f"{unpadded.group(1)}{unpadded.group(2)}")

    if 1 <= position <= len(peptide):
        return [f"{peptide[position - 1]}({name})" for name in names]
    tokens = [f"({name})" for name in names]
    if peptide:
        tokens.extend(f"{peptide[0]}({name})" for name in names)
    return tokens


def _mod_unimod_id(mod: dict, peptide: str) -> int:
    """Resolve a modification to its UNIMOD id, or -1 if unresolved.

    A unimod_id already set by the extractor is trusted; otherwise identity comes
    from LEGACY_PTM_TO_UNIMOD. Matching the delta mass against the tracked PTMs
    is a last resort, for when that table is unavailable or lacks the token.
    """
    uid = int(mod.get("unimod_id", -1))
    if uid > 0:
        return uid

    tokens = _legacy_ptm_tokens(
        str(mod.get("mod_name", "")), int(mod.get("position", 0)), peptide,
    )
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
) -> None:
    """Apply spectrum-level labels to one token, in place.

    Spectrum-level concepts are replicated to every token of a spectrum, noise
    peaks included. OR semantics so a future name collision cannot switch off a
    peak-level label that was already true.
    """
    for cname, val in spectrum_concepts.items():
        token_concepts[cname] = bool(token_concepts.get(cname, False)) or bool(val)


def _mod_to_ptm_name(mod: dict, peptide: str) -> str | None:
    """Map a modification to a tracked PTM concept name, or None if untracked."""
    entry = TRACKED_PTMS.get(_mod_unimod_id(mod, peptide))
    return entry[0] if entry else None


def exact_mass_proforma(proforma: str) -> str:
    """Substitute the dataset's rounded delta masses with exact PTM masses.

    At a ppm tolerance the dataset's two-decimal rounding can drop a modified
    fragment out of the match window. UNIMOD ProForma would fix it but
    spectrum_utils resolves UNIMOD ids over the network, so each rounded delta
    matching a tracked PTM by mass is rewritten to its exact monoisotopic mass
    instead. UNIMOD tags and unrecognised deltas are left untouched.
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

    return _PROFORMA_TAG_RE.sub(repl, proforma)


# --- spectrum_utils adapter (0.5.x) -------------------------------------------
# Each peak carries a PeakInterpretation holding FragmentAnnotation objects, each
# with a combined ion_type string ("b2", "y3", "IC" for the Cys immonium, "m3:5"
# for an internal fragment, "p" for precursor), plus charge, neutral_loss,
# isotope and mz_delta. The only code that depends on that layout.
_BACKBONE_RE = re.compile(r"^([abcxyz])(\d+)$")
_INTERNAL_RE = re.compile(r"^m(\d+):(\d+)$")
_IMMONIUM_RE = re.compile(r"^I([A-Z])")

# Preference order when a peak has several candidate annotations (lower = better).
_ION_PRIORITY = {"b": 0, "y": 0, "a": 1, "c": 1, "x": 1, "z": 1, "I": 2, "internal": 3, "precursor": 4}

# charge 0 means "no fragment charge", which is what an unmatched peak has. It
# must not be 1: _set_physical_concepts reads this field directly, so a noise
# sentinel of 1 would label every unmatched peak is_fragment_charge_1 -- pushing
# that concept's base rate to ~0.96 and making it fire mostly on noise.
_NOISE = {"ion_type": "noise", "ion_index": None, "charge": 0, "neutral_loss": ""}


def _parse_ion_type(ion_type_str: str) -> tuple:
    """Parse a spectrum_utils ion_type string into (type, index).

    index is an int for backbone ions, a (start, end) tuple for internal
    fragments, the residue letter for immonium ions, None for the precursor, and
    the whole result is (None, None) for anything unrecognised.
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


def _candidate_sort_key(fa, ion_type: str, neutral_loss: str, charge: int) -> tuple:
    """Preference ordering for competing annotations of one peak (lower = better).

    Monoisotopic before isotopic, no neutral loss before a loss, canonical b/y
    before other series, then lowest charge, then smallest mass error -- the
    annotation most consistent with the observed mass wins.
    """
    isotope = int(getattr(fa, "isotope", 0) or 0)
    mz_delta = getattr(fa, "mz_delta", None)
    return (
        isotope != 0,
        bool(neutral_loss),
        _ION_PRIORITY.get(ion_type, 5),
        charge,
        abs(mz_delta[0]) if mz_delta else 0.0,
    )


def _annotation_to_dict(peak_interpretation) -> dict:
    """Reduce one PeakInterpretation to {ion_type, ion_index, charge,
    neutral_loss}, picking the best candidate (see _candidate_sort_key).
    Unannotated peaks become noise."""
    frags = getattr(peak_interpretation, "fragment_annotations", None) or []
    candidates = []
    for fa in frags:
        ion_type, ion_index = _parse_ion_type(getattr(fa, "ion_type", ""))
        if ion_type is None:
            continue
        nl_raw = getattr(fa, "neutral_loss", None)
        neutral_loss = str(nl_raw).lstrip("-") if nl_raw else ""
        charge = int(getattr(fa, "charge", 1))
        candidates.append((
            _candidate_sort_key(fa, ion_type, neutral_loss, charge),
            {
                "ion_type": ion_type,
                "ion_index": ion_index,
                "charge": charge,
                "neutral_loss": neutral_loss,
            },
        ))

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
            neutral_losses=NEUTRAL_LOSSES,
        )
    except Exception as exc:
        # Bad ProForma or unsupported residue: all-noise keeps the run going.
        LOG.warning("spectrum_utils annotation failed for %r: %s", proforma, exc)
        return [dict(_NOISE) for _ in range(len(mz_np))]

    annotations = getattr(spectrum, "annotation", None)
    if annotations is None:
        return [dict(_NOISE) for _ in range(len(mz_np))]
    return [_annotation_to_dict(a) for a in annotations]


# --- Concept computation ------------------------------------------------------

def compute_spectrum_concepts(
    modifications: list[dict],
    precursor_charge: int,
    peptide: str,
) -> dict[str, bool]:
    """Concepts that depend only on the spectrum, not the current peak. These
    are replicated across all of its tokens."""
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
    """Intensity threshold for the top-decile concept, computed per spectrum so
    prevalence stays calibrated across differing dynamic ranges."""
    if intensities.numel() == 0:
        return float("inf")
    # 90th percentile: top 10% of peaks fire above this.
    return float(torch.quantile(intensities.float(), 0.9))


def _set_ion_type_concepts(out: dict[str, bool], ion_type: str) -> None:
    """Ion-identity family. is_matched_peak is the complement of noise/latent."""
    out["is_b_ion"] = ion_type == "b"
    out["is_y_ion"] = ion_type == "y"
    out["is_I_ion"] = ion_type == "I"
    out["is_internal_fragment"] = ion_type == "internal"
    out["is_precursor_related"] = ion_type == "precursor"
    out["is_noise_peak"] = ion_type == "noise"
    out["is_latent_token"] = False  # latent handled separately
    out["is_matched_peak"] = ion_type not in ("noise", "latent")


def _set_neutral_loss_concepts(out: dict[str, bool], neutral_loss: str) -> None:
    nl = neutral_loss or ""
    out["is_H2O_loss"] = "H2O" in nl
    out["is_NH3_loss"] = "NH3" in nl
    out["has_neutral_loss"] = bool(nl)


def _set_physical_concepts(
    out: dict[str, bool],
    mz: float,
    intensity: float,
    intensity_threshold: float,
    fragment_charge: int,
) -> None:
    """Properties readable from the peak itself: charge, m/z region, intensity.

    m/z and intensity are true of any peak, matched or not. Fragment charge is
    not: it comes from the annotation, so an unmatched peak carries charge 0
    (see _NOISE) and both charge concepts are correctly False for it.
    """
    out["is_fragment_charge_1"] = fragment_charge == 1
    out["is_fragment_charge_2"] = fragment_charge == 2

    low_cut, high_cut = MZ_LOW_HIGH_CUT
    out["mz_low"] = mz < low_cut
    out["mz_mid"] = low_cut <= mz < high_cut
    out["mz_high"] = mz >= high_cut

    out["top_decile_intensity"] = intensity >= intensity_threshold


def _set_position_concepts(
    out: dict[str, bool], geom: IonGeometry, peptide_length: int
) -> None:
    """Position and first/last-ion families, defined only for backbone b/y ions.

    These two families deliberately use different coordinates for y ions, so a
    y_1 token is both position_Cterm and is_first_ion. That is not a
    contradiction -- they measure different axes:

    position_*      where along the backbone the cleavage happened. y_k cleaves
                    after residue L-k, so k is converted to a cleavage site
                    before binning into thirds; b_k already cleaves after k.

    is_first/last   how far along its own ion ladder the fragment sits. Index 1
                    is the smallest ion of either series (b_1, y_1) and L-1 the
                    largest, which keeps each concept spectrally coherent: b_1
                    and y_1 are both light, b_{L-1} and y_{L-1} both heavy.
                    Defining these by cleavage site instead would pair b_1 with
                    y_{L-1} -- a light peak and a heavy one under one label.
    """
    k = geom.backbone_index()
    if k is None:
        return

    if k > 0:
        cleavage_pos = k if geom.ion_type == "b" else (peptide_length - k)
        third = peptide_length / 3.0
        out["position_Nterm"] = cleavage_pos < third
        out["position_middle"] = third <= cleavage_pos < 2 * third
        out["position_Cterm"] = cleavage_pos >= 2 * third

    out["is_first_ion"] = k == 1
    out["is_last_ion"] = k == peptide_length - 1


def _cleavage_site(geom: IonGeometry, peptide_length: int) -> int | None:
    """1-indexed residue after which this ion's backbone cleavage occurred.

    b_k cleaves after residue k, its complementary y_k after residue L-k. None
    for non-backbone ions, or indices implying no interior cleavage.
    """
    k = geom.backbone_index()
    if k is None or not (1 <= k < peptide_length):
        return None
    return k if geom.ion_type == "b" else peptide_length - k


def _set_cleavage_concepts(
    out: dict[str, bool], geom: IonGeometry, peptide: str, peptide_length: int
) -> None:
    """Canonical (tryptic) and enhanced cleavage families.

    Named for the residue flanking the break: C_to_X means X sits before it,
    N_to_Pro means proline sits after it.
    """
    site = _cleavage_site(geom, peptide_length)
    if site is None or not (1 <= site < peptide_length):
        return

    before = peptide[site - 1]
    after = peptide[site]
    out["cleaves_C_to_Lys"] = (before == "K")
    out["cleaves_C_to_Arg"] = (before == "R")
    out["cleaves_C_to_Asp"] = (before == "D")
    out["cleaves_C_to_Glu"] = (before == "E")
    out["cleaves_N_to_Pro"] = (after == "P")


def _set_residue_cover_concepts(
    out: dict[str, bool], geom: IonGeometry, peptide: str
) -> None:
    # Every covers_<residue> name is registered (ConceptRegistry._check_derived_names).
    for residue in COVERED_RESIDUES:
        out[f"covers_{residue}"] = geom.covers_any_residue(peptide, residue)


def _set_ion_ptm_concepts(
    out: dict[str, bool],
    geom: IonGeometry,
    peptide: str,
    modifications: list[dict],
    peptide_length: int,
) -> None:
    """Token-level PTM containment.

    A b/y/internal ion that spans the modified residue's position contains it;
    immonium ions don't localise (geom.covers returns False for them), which is
    what makes ion_contains_* a genuine localisation signal rather than a
    spectrum-level correlate.
    """
    for mod in modifications:
        mod_pos = mod.get("position", 0)
        if mod_pos < 1 or mod_pos > peptide_length:
            continue
        ptm_name = _mod_to_ptm_name(mod, peptide)
        if ptm_name is None:
            continue
        if geom.covers(mod_pos):
            out[f"ion_contains_{ptm_name}"] = True


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
    geom = IonGeometry(ion_type, annotation["ion_index"], peptide_length)

    _set_ion_type_concepts(out, ion_type)
    _set_neutral_loss_concepts(out, annotation["neutral_loss"])
    _set_physical_concepts(
        out, mz, intensity, intensity_threshold, annotation["charge"],
    )
    _set_position_concepts(out, geom, peptide_length)
    _set_cleavage_concepts(out, geom, peptide, peptide_length)
    _set_residue_cover_concepts(out, geom, peptide)
    _set_ion_ptm_concepts(out, geom, peptide, modifications, peptide_length)

    return out


def compute_latent_concepts(
    spectrum_concepts: dict[str, bool],
    peptide: str,
    modifications: list[dict],
    registry: ConceptRegistry,
) -> dict[str, bool]:
    """Concept dict for the latent token at position 0.

    The latent covers the whole peptide, so residue-cover and token-PTM concepts
    apply if the residue or modification exists anywhere in it. Peak-specific
    concepts (m/z bin, ion type, neutral loss, charge) stay False.
    """
    out: dict[str, bool] = {n: False for n in registry.names}
    merge_spectrum_concepts(out, spectrum_concepts)
    out["is_latent_token"] = True

    # Latent covers whole peptide for residue-cover concepts.
    for residue in COVERED_RESIDUES:
        out[f"covers_{residue}"] = residue in peptide

    # Latent contains all PTMs the spectrum contains.
    for mod in modifications:
        ptm_name = _mod_to_ptm_name(mod, peptide)
        if ptm_name is None:
            continue
        out[f"ion_contains_{ptm_name}"] = True

    return out


# --- Per-chunk annotation -----------------------------------------------------

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
    num_workers: int = 1                # parallel chunk-annotation processes; see AnnotationRunner

    def as_jsonable(self) -> dict:
        out = dataclasses.asdict(self)
        out["extract_dir"] = str(self.extract_dir)
        out["output_dir"] = str(self.output_dir)
        return out


class _SpectrumTokens:
    """Per-token tensors for one spectrum: latent token at row 0, then one row
    per peak. Allocated all-zero and filled row by row, which is cheaper than
    indexing into a chunk-wide tensor from Python.
    """

    def __init__(self, n_tokens: int, n_concepts: int):
        self.labels = torch.zeros(n_tokens, n_concepts, dtype=torch.bool)
        self.ion_type = torch.zeros(n_tokens, dtype=torch.int8)
        self.ion_index = torch.full((n_tokens,), -1, dtype=torch.int16)
        self.fragment_charge = torch.zeros(n_tokens, dtype=torch.int8)
        self.neutral_loss_id = torch.zeros(n_tokens, dtype=torch.int8)
        self.mz = torch.zeros(n_tokens, dtype=torch.float32)
        self.intensity = torch.zeros(n_tokens, dtype=torch.float32)

    def set_concepts(self, row: int, concepts: dict[str, bool], name_to_idx: dict[str, int]) -> None:
        for cname, val in concepts.items():
            if val:
                self.labels[row, name_to_idx[cname]] = True

    def set_peak_metadata(self, row: int, ann: dict, mz: float, intensity: float) -> None:
        """Store the compact per-token metadata for one annotated peak."""
        self.ion_type[row] = ION_TYPE_VOCAB.get(ann["ion_type"], 0)

        ii = ann["ion_index"]
        if isinstance(ii, int) and ii >= 0:
            self.ion_index[row] = min(ii, _ION_INDEX_CAP)
        elif isinstance(ii, tuple):
            # Internal fragment: store the start position only.
            self.ion_index[row] = min(ii[0], _ION_INDEX_CAP)

        self.fragment_charge[row] = min(ann["charge"], _FRAGMENT_CHARGE_CAP)
        self.neutral_loss_id[row] = NEUTRAL_LOSS_VOCAB.get(ann["neutral_loss"] or "", 0)
        self.mz[row] = mz
        self.intensity[row] = intensity


def _annotate_spectrum(
    chunk_data: dict,
    s: int,
    config: AnnotationConfig,
    registry: ConceptRegistry,
    name_to_idx: dict[str, int],
) -> _SpectrumTokens:
    """Annotate spectrum `s` of a chunk into its per-token tensors, ordered as
    extract.py flattened it: the latent summary token, then one token per peak.
    """
    peptide = chunk_data["peptides"][s]
    modifications = chunk_data["modifications"][s]
    mz_array = chunk_data["mz_arrays"][s]
    intensity_array = chunk_data["intensity_arrays"][s]
    precursor_charge = int(chunk_data["precursor_charges"][s])
    peptide_length = len(peptide)

    spectrum_concepts = compute_spectrum_concepts(
        modifications, precursor_charge, peptide,
    )
    intensity_threshold = compute_top_decile_threshold(intensity_array)

    # One label dict per peak (noise if unmatched).
    annotations = run_spectrum_utils(
        proforma=chunk_data["proforma_strings"][s],
        mz_array=mz_array,
        intensity_array=intensity_array,
        precursor_mz=float(chunk_data["precursor_mzs"][s]),
        precursor_charge=precursor_charge,
        ion_types=config.ion_types,
        fragment_tol_mass=config.fragment_tol_mass,
        fragment_tol_mode=config.fragment_tol_mode,
        enable_internal=config.enable_internal,
    )

    tokens = _SpectrumTokens(1 + len(mz_array), len(registry.names))

    # Row 0: the latent summary token.
    tokens.set_concepts(
        0,
        compute_latent_concepts(
            spectrum_concepts, peptide, modifications, registry,
        ),
        name_to_idx,
    )
    tokens.ion_type[0] = ION_TYPE_VOCAB["latent"]

    # Rows 1..n_peaks: one per spectral peak.
    for peak_idx, ann in enumerate(annotations):
        row = peak_idx + 1
        mz = float(mz_array[peak_idx])
        intensity = float(intensity_array[peak_idx])

        peak_concepts = compute_peak_concepts(
            ann, mz, intensity, intensity_threshold,
            peptide, modifications, peptide_length, registry,
        )
        # Spectrum-level concepts apply to every peak too. Merge with OR
        # semantics so spectrum labels cannot clobber peak-local labels if a
        # future concept family reuses a name.
        merge_spectrum_concepts(peak_concepts, spectrum_concepts)

        tokens.set_concepts(row, peak_concepts, name_to_idx)
        tokens.set_peak_metadata(row, ann, mz, intensity)

    return tokens


def annotate_chunk(
    chunk_data: dict,
    config: AnnotationConfig,
    registry: ConceptRegistry,
) -> LabelChunkData:
    """Build a LabelChunkData for one input chunk.

    `chunk_data` is a ChunkMeta loaded as a dict. Token order matches the
    chunk's flattening -- per spectrum, the latent token then one token per peak
    -- so the result row-aligns with activations[layer] for that chunk.
    """
    n_spectra = chunk_data["n_spectra"]
    name_to_idx = registry.index

    per_spectrum = [
        _annotate_spectrum(chunk_data, s, config, registry, name_to_idx)
        for s in range(n_spectra)
    ]

    def joined(attr: str) -> torch.Tensor:
        return torch.cat([getattr(t, attr) for t in per_spectrum], dim=0)

    token_labels = joined("labels")

    # Row alignment with the activations is this module's whole contract, and
    # nothing downstream can detect a violation: the labels would simply
    # describe the wrong tokens. The extractor recorded its own token count, so
    # compare against it here rather than trusting the two to agree.
    expected_tokens = int(chunk_data["total_tokens"])
    if int(token_labels.size(0)) != expected_tokens:
        raise ValueError(
            f"Chunk {chunk_data['chunk_idx']}: annotated {int(token_labels.size(0))} "
            f"tokens but extraction recorded {expected_tokens}. Labels would not "
            "row-align with activations. This means the peak arrays in the chunk "
            "metadata disagree with the token count extraction derived from the "
            "spectra mask."
        )

    return LabelChunkData(
        schema_version=ANNOTATION_SCHEMA_VERSION,
        chunk_idx=chunk_data["chunk_idx"],
        n_spectra=n_spectra,
        total_tokens=int(token_labels.size(0)),
        concept_names=list(registry.names),
        token_labels=token_labels,
        ion_type_ids=joined("ion_type"),
        ion_indices=joined("ion_index"),
        fragment_charges=joined("fragment_charge"),
        neutral_loss_ids=joined("neutral_loss_id"),
        peak_mzs=joined("mz"),
        peak_intensities=joined("intensity"),
    )


def _annotate_and_save_chunk(
    extract_dir: Path,
    meta_rel_path: str,
    config: AnnotationConfig,
    registry: ConceptRegistry,
    out_path: Path,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Annotate one chunk, save it, and return its partial co-occurrence stats.

    Returning stats rather than the label tensor keeps what crosses the process
    boundary small; folding them in is a pure `+=` (see add_partial), so doing
    it per chunk is safe. Module-level, with plain-data arguments, so
    ProcessPoolExecutor can pickle it to a worker.

    Returns (chunk_idx, n_tokens, cooccur, marginal).
    """
    path = extract_dir / meta_rel_path
    chunk_data = torch.load(path, map_location="cpu", weights_only=False)
    label_chunk = annotate_chunk(chunk_data, config, registry)
    label_chunk.save(out_path)

    x = label_chunk.token_labels.to(torch.float64)
    cooccur = x.T @ x
    marginal = x.sum(dim=0)
    return chunk_data["chunk_idx"], label_chunk.total_tokens, cooccur, marginal


class ConceptStatsAccumulator:
    """Accumulate co-occurrence counts across chunks; the phi-coefficient matrix
    is derived from them at the end of the run."""

    def __init__(self, n_concepts: int):
        self.n_concepts = n_concepts
        self.cooccur = torch.zeros((n_concepts, n_concepts), dtype=torch.float64)
        self.marginal = torch.zeros(n_concepts, dtype=torch.float64)
        self.total = 0

    def add_chunk(self, labels_bool: torch.Tensor) -> None:
        """Update accumulators with one chunk's [n_tokens, n_concepts] labels."""
        x = labels_bool.to(torch.float64)
        self.add_partial(x.size(0), x.T @ x, x.sum(dim=0))

    def add_partial(
        self, n_tokens: int, cooccur: torch.Tensor, marginal: torch.Tensor,
    ) -> None:
        """Fold in one chunk's already-computed co-occurrence/marginal/count.

        A pure `+=`, so order does not matter -- which is what lets chunks be
        annotated in parallel and still produce output identical to the
        sequential, in-order loop.
        """
        self.cooccur += cooccur
        self.marginal += marginal
        self.total += n_tokens

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
        self.extract_manifest = self._load_extract_manifest(config.extract_dir)

        LOG.info("Annotation registry: %d concepts across %d families",
                 len(self.registry.names), len(CONCEPT_FAMILIES))

    @staticmethod
    def _load_extract_manifest(extract_dir: Path) -> dict:
        """Load the extraction manifest, rejecting an incompatible schema.

        Labels must row-align with the activations they are joined against, so a
        manifest from a different extraction layout is fatal, not a warning.
        """
        path = extract_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Extract manifest not found at {path}")
        manifest = json.loads(path.read_text())
        if manifest["schema_version"] != EXTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"Extraction schema mismatch: manifest={manifest['schema_version']}, "
                f"annotator expects {EXTRACT_SCHEMA_VERSION}. Re-run extract.py."
            )
        return manifest

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self.labels_dir / f"chunk_{chunk_idx:05d}.pt"

    def run(self) -> None:
        accumulator = ConceptStatsAccumulator(len(self.registry.names))
        n_chunks = self.extract_manifest["n_chunks"]
        t0 = time.time()

        to_annotate: list[int] = []
        for chunk_idx in range(n_chunks):
            out_path = self._chunk_path(chunk_idx)
            if self.config.resume and out_path.exists():
                # Re-load existing labels so accumulator picks them up for stats.
                LOG.info("Chunk %d already annotated, loading for stats", chunk_idx)
                existing = LabelChunkData.load(out_path)
                if existing.schema_version != ANNOTATION_SCHEMA_VERSION:
                    raise ValueError(
                        f"Annotation schema mismatch in {out_path}: "
                        f"chunk={existing.schema_version}, annotator expects "
                        f"{ANNOTATION_SCHEMA_VERSION}. Re-run with --no-resume "
                        "(or clear the output dir) to regenerate."
                    )
                # The schema version does not change when a concept is added to
                # the registry, so compare the concept set too: otherwise stale
                # chunks fold into the accumulator at the wrong width (an opaque
                # matmul shape error) or, if the widths happen to match, silently
                # under a different concept ordering.
                if existing.concept_names != self.registry.names:
                    raise ValueError(
                        f"Concept registry mismatch in {out_path}: chunk has "
                        f"{len(existing.concept_names)} concepts, annotator has "
                        f"{len(self.registry.names)}. Re-run with --no-resume "
                        "(or clear the output dir) to regenerate."
                    )
                accumulator.add_chunk(existing.token_labels)
            else:
                to_annotate.append(chunk_idx)

        if to_annotate:
            self._annotate_chunks(to_annotate, accumulator, t0)

        self._write_manifest(accumulator, n_chunks)
        LOG.info(
            "Annotation complete: %d chunks, %d tokens, %.0fs",
            n_chunks, accumulator.total, time.time() - t0,
        )

    def _annotate_chunks(
        self, chunk_indices: list[int], accumulator: ConceptStatsAccumulator, t0: float,
    ) -> None:
        """Annotate each chunk and fold its stats into the accumulator.

        Sequential in this process when num_workers <= 1, otherwise fanned out
        across a process pool. Both paths call the same annotate_chunk() and
        write the same files; worker count changes wall-clock time only, never
        output, because add_partial is order-independent.
        """
        n_total = len(chunk_indices)

        if self.config.num_workers <= 1:
            for i, chunk_idx in enumerate(chunk_indices, start=1):
                meta_rel = self.extract_manifest["chunks"][chunk_idx]["meta"]
                out_path = self._chunk_path(chunk_idx)
                got_idx, n_tokens, cooccur, marginal = _annotate_and_save_chunk(
                    self.config.extract_dir, meta_rel, self.config, self.registry, out_path,
                )
                assert got_idx == chunk_idx, f"chunk index desync: expected {chunk_idx}, got {got_idx}"
                accumulator.add_partial(n_tokens, cooccur, marginal)
                LOG.info(
                    "  Annotated chunk %d (%d/%d) | n_tokens=%d | elapsed=%.1fs",
                    chunk_idx, i, n_total, n_tokens, time.time() - t0,
                )
            return

        LOG.info(
            "Annotating %d chunk(s) using up to %d worker process(es)",
            n_total, self.config.num_workers,
        )
        with ProcessPoolExecutor(
            max_workers=self.config.num_workers, initializer=_init_annotation_worker,
        ) as pool:
            futures = {
                pool.submit(
                    _annotate_and_save_chunk,
                    self.config.extract_dir,
                    self.extract_manifest["chunks"][chunk_idx]["meta"],
                    self.config,
                    self.registry,
                    self._chunk_path(chunk_idx),
                ): chunk_idx
                for chunk_idx in chunk_indices
            }
            completed = 0
            for future in as_completed(futures):
                chunk_idx = futures[future]
                got_idx, n_tokens, cooccur, marginal = future.result()
                assert got_idx == chunk_idx, f"chunk index desync: expected {chunk_idx}, got {got_idx}"
                accumulator.add_partial(n_tokens, cooccur, marginal)
                completed += 1
                LOG.info(
                    "  Annotated chunk %d (%d/%d) | n_tokens=%d | elapsed=%.1fs",
                    chunk_idx, completed, n_total, n_tokens, time.time() - t0,
                )

    def _write_manifest(self, accumulator: ConceptStatsAccumulator, n_chunks: int) -> None:
        base_rates = accumulator.base_rates().tolist()

        manifest = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
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
        atomic_torch_save({
            "concept_names": self.registry.names,
            "phi": accumulator.phi_matrix(),
            "marginal": accumulator.marginal.to(torch.float32),
            "cooccur": accumulator.cooccur.to(torch.float32),
            "n_tokens": int(accumulator.total),
        }, phi_path)
        LOG.info("Wrote manifest to %s and phi matrix to %s", self.manifest_path, phi_path)


# --- CLI ----------------------------------------------------------------------

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
    p.add_argument("--num-workers", type=int, default=None,
                   help="Parallel worker processes for chunk annotation. "
                        "Worker count affects wall-clock time only, never "
                        "output. Defaults to os.cpu_count(); annotation is "
                        "CPU-only, so this uses cores that would otherwise idle "
                        "while the GPU works elsewhere. Pass 1 for sequential.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    num_workers = args.num_workers if args.num_workers is not None else (os.cpu_count() or 1)
    config = AnnotationConfig(
        extract_dir=args.extract_dir,
        output_dir=args.output_dir,
        ion_types=args.ion_types,
        enable_internal=not args.no_internal,
        fragment_tol_mass=args.fragment_tol,
        fragment_tol_mode=args.fragment_tol_mode,
        resume=not args.no_resume,
        num_workers=num_workers,
    )

    runner = AnnotationRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
