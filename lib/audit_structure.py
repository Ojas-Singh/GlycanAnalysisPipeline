"""Read a glycan back from its coordinates, for the entry audit (``lib/audit.py``).

The audit has to say whether the molecule that was simulated is the molecule
the entry is named after.  The residue names of a structure file cannot say
that: they are what the file was told it holds.  The coordinates can.  This
module reads, from the atoms of one frame alone (element, atom name, residue
number and position; no residue name, no bond record):

* the sugar residues: a ring of five or six atoms with one oxygen, inside one
  residue;
* for every sugar residue its carbon chain (``C1``, ``C2`` ... by atom name,
  each bonded to the next), the anomeric carbon (the ring carbon next to the
  ring oxygen that carries a second, exocyclic oxygen) and the ring size;
* the configuration of every stereocentre of the chain as the side of its
  heteroatom in the Fischer projection, and so the WURCS skeleton code of the
  residue (``a2122h`` for a glucopyranose: ``a`` the anomeric carbon, ``2``
  right, ``1`` left, ``d`` deoxy, ``h`` CH2-O, ``m`` CH3, ``A`` carboxyl);
* the anomer, alpha when the exocyclic oxygen of the anomeric carbon is on the
  same side of the Fischer projection as the heteroatom of the anomeric
  reference atom (IUPAC 2-Carb-6.2), the reference atom being the
  highest-numbered of the first four stereocentres after the anomeric carbon
  (C5 of a hexose, C4 of a pentose, C7 of a sialic acid);
* the glycosidic linkages: donor, its anomeric position, acceptor and the
  position of the acceptor carbon whose oxygen carries the donor.

**The side of a heteroatom.**  In a Fischer projection the chain is vertical,
the lower-numbered neighbour up and the higher-numbered one down, both
pointing away from the viewer, and the two horizontal substituents point
towards him.  For a centre ``c`` with chain neighbours ``u`` (up) and ``d``
(down) and heteroatom ``x``, the signed volume ``(u - c) . ((d - c) x (x - c))``
is negative when ``x`` is on the right and positive when it is on the left.
For the anomeric carbon, "up" is the ring oxygen: the cyclic (Tollens)
projection draws the ring bond of the anomeric carbon vertically, above C1 of
an aldose and in place of C1 of a ketose, which then stands beside the
anomeric carbon.  (Checked against crabWURCS: the stereo SMILES it writes for
alpha and beta D-Glcp, L-Fucp, L-Araf, D-Galf, D-Fruf, D-Frup, Neu5Ac and Kdn,
made three-dimensional by Open Babel, read back with the anomer of the name;
and against the residue codes of the GLYCAM structures of the raw archive.)

Bonds are perceived from distances with the rule of the frame-quality filter
(``lib/frame_quality.py``: two atoms are bonded when they are closer than the
sum of their covalent radii plus 0.35 angstrom; no H-H bond; only the element
pairs that can bond).  The same signed volume is the configuration test of the
stereocentre analysis of the thesis (``robustness_data.py``, ``stereocentres``),
which :func:`stereocentres` ports for the comparison of the runs of an entry.

What is compared with a name (:func:`compare`) is the WURCS of that name, as
crabWURCS writes it: the skeleton code, the anomer and the ring of every
residue, and every linkage.  Substituents (N-acetyl, sulfate, methyl) are not
part of a skeleton code and are not compared here; the audit counts them
through the mass of the structure.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

#: Covalent radii and the element pairs that may bond: those of ``lib/frame_quality.py``.
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
                  "CL": 1.02, "BR": 1.20, "I": 1.39}
BOND_TOLERANCE = 0.35
_ALLOWED = {frozenset(p) for p in (("H", "C"), ("H", "N"), ("H", "O"), ("H", "S"), ("C", "C"), ("C", "N"),
                                   ("C", "O"), ("C", "S"), ("N", "N"), ("N", "O"), ("N", "S"), ("O", "P"),
                                   ("O", "S"), ("P", "S"), ("S", "S"))}
HETEROATOMS = ("O", "N", "S")
_CARBON_NAME = re.compile(r"^C(\d+)$")
_METHYL_NAME = re.compile(r"^C(\d+)M$")


@dataclass
class Atoms:
    """Atoms of one frame: names, residue names and numbers, elements and coordinates (angstrom)."""

    names: List[str]
    residue_names: List[str]
    residue_keys: List[Tuple[str, int, str]]       # (chain, number, insertion code): one residue
    elements: List[str]
    xyz: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def with_coordinates(self, xyz: np.ndarray) -> "Atoms":
        return Atoms(self.names, self.residue_names, self.residue_keys, self.elements, np.asarray(xyz, dtype=np.float64))


def element_of(line: str) -> str:
    """Element of a PDB ATOM/HETATM line: columns 77-78, else the first letter of the atom name."""
    symbol = line[76:78].strip().upper() if len(line) >= 78 else ""
    if symbol:
        return symbol
    return line[12:16].strip().lstrip("0123456789")[:1].upper()


def atoms_from_lines(lines: Iterable[str]) -> Atoms:
    names, rnames, keys, elements, xyz = [], [], [], [], []
    for line in lines:
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        names.append(line[12:16].strip())
        rnames.append(line[17:21].strip())
        keys.append((line[21:22], int(line[22:26]), line[26:27]))
        elements.append(element_of(line))
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return Atoms(names, rnames, keys, elements, np.asarray(xyz, dtype=np.float64).reshape(-1, 3))


def read_first_model(path: Union[str, Path]) -> Atoms:
    """ATOM/HETATM records of the first model of a PDB file (all of it when there is no MODEL record)."""
    lines = []
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if line[:6] in ("ATOM  ", "HETATM"):
                lines.append(line)
    return atoms_from_lines(lines)


def neighbours(atoms: Atoms) -> List[List[int]]:
    """Bonded atoms of every atom, from the distances of this frame (the rule of the frame-quality filter)."""
    n = len(atoms)
    el = [e if e in COVALENT_RADII else "" for e in atoms.elements]
    radius = np.array([COVALENT_RADII.get(e, 0.0) for e in el])
    d = np.linalg.norm(atoms.xyz[:, None, :] - atoms.xyz[None, :, :], axis=2)
    close = d < radius[:, None] + radius[None, :] + BOND_TOLERANCE
    np.fill_diagonal(close, False)
    out: List[List[int]] = [[] for _ in range(n)]
    for i, j in zip(*np.where(np.triu(close, 1))):
        if el[i] and el[j] and frozenset((el[i], el[j])) in _ALLOWED:
            out[int(i)].append(int(j))
            out[int(j)].append(int(i))
    return out


def signed_volume(xyz: np.ndarray, centre: int, a: int, b: int, c: int) -> float:
    """``(a - centre) . ((b - centre) x (c - centre))``."""
    o = xyz[centre]
    return float(np.dot(xyz[a] - o, np.cross(xyz[b] - o, xyz[c] - o)))


# ---------------------------------------------------------------------------
# Residues
# ---------------------------------------------------------------------------
@dataclass
class SugarResidue:
    """A sugar residue read from the atoms."""

    key: Tuple[str, int, str]
    residue_name: str
    number: int
    atoms: List[int]
    ring: List[int]                       # ring atoms
    ring_oxygen: int
    anomeric: Optional[int] = None        # atom index of the anomeric carbon
    anomeric_position: Optional[int] = None
    chain: Dict[int, int] = field(default_factory=dict)   # carbon number -> atom index
    exocyclic: Optional[int] = None       # the exocyclic heteroatom of the anomeric carbon
    skeleton: Optional[str] = None        # WURCS skeleton code
    ring_close: Optional[int] = None      # number of the ring carbon bonded to the ring oxygen (not anomeric)
    anomer: Optional[str] = None          # "a", "b" or None (undetermined)
    anomeric_volume: Optional[float] = None
    reference_position: Optional[int] = None
    reference_volume: Optional[float] = None
    problem: Optional[str] = None         # why the residue could not be read completely

    @property
    def ring_code(self) -> Optional[str]:
        if self.anomeric_position is None or self.ring_close is None:
            return None
        return f"{self.anomeric_position}-{self.ring_close}"

    @property
    def code(self) -> Optional[str]:
        """Skeleton, anomer and ring as WURCS writes a unique residue (``a2122h-1b_1-5``)."""
        if self.skeleton is None:
            return None
        anomer = self.anomer or "x"
        return f"{self.skeleton}-{self.anomeric_position}{anomer}_{self.ring_code}"

    def describe(self) -> Dict[str, Any]:
        return {"residue": self.residue_name, "number": self.number, "skeleton": self.skeleton,
                "anomer": self.anomer, "ring": self.ring_code, "code": self.code,
                "reference_carbon": f"C{self.reference_position}" if self.reference_position else None,
                "problem": self.problem}


def _residue_groups(atoms: Atoms) -> Dict[Tuple[str, int, str], List[int]]:
    groups: Dict[Tuple[str, int, str], List[int]] = {}
    for i, key in enumerate(atoms.residue_keys):
        groups.setdefault(key, []).append(i)
    return groups


def _ring_through(o: int, members: set, nb: List[List[int]], elements: List[str]) -> Optional[List[int]]:
    """The smallest ring of 5 or 6 heavy atoms of one residue through the oxygen ``o``, or None."""
    carbons = [j for j in nb[o] if j in members and elements[j] == "C"]
    best = None
    for i in range(len(carbons)):
        for k in range(i + 1, len(carbons)):
            start, goal = carbons[i], carbons[k]
            # breadth-first path start -> goal inside the residue, heavy atoms, not through o
            previous = {start: None}
            queue = [start]
            while queue and goal not in previous:
                nxt = []
                for a in queue:
                    for b in nb[a]:
                        if b in previous or b == o or b not in members or elements[b] == "H":
                            continue
                        previous[b] = a
                        nxt.append(b)
                queue = nxt
            if goal not in previous:
                continue
            path = [goal]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            ring = [o] + path
            if len(ring) in (5, 6) and sum(elements[a] == "O" for a in ring) == 1 and \
                    (best is None or len(ring) < len(best)):
                best = ring
    return best


def _fischer(xyz: np.ndarray, centre: int, up: int, down: int, x: int) -> float:
    return signed_volume(xyz, centre, up, down, x)


def read_residues(atoms: Atoms, nb: Optional[List[List[int]]] = None) -> List[SugarResidue]:
    """The sugar residues of a structure, each with its skeleton code and anomer where they can be read."""
    nb = neighbours(atoms) if nb is None else nb
    el = atoms.elements
    out: List[SugarResidue] = []
    for key, members_list in _residue_groups(atoms).items():
        members = set(members_list)
        ring = None
        for o in members_list:
            if el[o] == "O":
                ring = _ring_through(o, members, nb, el)
                if ring:
                    break
        if not ring:
            continue
        ring_o = ring[0]
        residue = SugarResidue(key, atoms.residue_names[members_list[0]], key[1], members_list, ring, ring_o)
        out.append(residue)
        # carbons of the chain by name
        numbered = {}
        for i in members_list:
            m = _CARBON_NAME.match(atoms.names[i])
            if m and el[i] == "C":
                numbered[int(m.group(1))] = i
        for i in members_list:              # GLYCAM names the methyl carbon of some 6-deoxy residues C6M
            m = _METHYL_NAME.match(atoms.names[i])
            if m and el[i] == "C" and int(m.group(1)) not in numbered:
                numbered[int(m.group(1))] = i
        # anomeric carbon: a ring carbon next to the ring oxygen with an exocyclic heteroatom
        candidates = []
        for c in nb[ring_o]:
            if c not in members or el[c] != "C":
                continue
            exo = [x for x in nb[c] if el[x] in HETEROATOMS and x != ring_o and x not in ring]
            if exo:
                candidates.append((c, exo))
        name_of = {i: n for n, i in numbered.items()}
        candidates = [(c, exo) for c, exo in candidates if c in name_of]
        if len(candidates) != 1:
            residue.problem = "no single anomeric carbon" if candidates == [] else "two candidate anomeric carbons"
            if len(candidates) > 1:
                candidates.sort(key=lambda item: name_of[item[0]])
                candidates = candidates[:1]
            else:
                continue
        anomeric, exo = candidates[0]
        a = name_of[anomeric]
        residue.anomeric, residue.anomeric_position = anomeric, a
        residue.exocyclic = exo[0]
        # the chain: C1, C2 ... each bonded to the next, through the anomeric carbon
        chain = {a: anomeric}
        k = a
        while k - 1 in numbered and numbered[k - 1] in nb[chain[k]]:
            chain[k - 1] = numbered[k - 1]
            k -= 1
        k = a
        while k + 1 in numbered and numbered[k + 1] in nb[chain[k]]:
            chain[k + 1] = numbered[k + 1]
            k += 1
        residue.chain = chain
        other = [c for c in nb[ring_o] if c in members and el[c] == "C" and c != anomeric]
        if other and other[0] in name_of:
            residue.ring_close = name_of[other[0]]
        first, last = min(chain), max(chain)
        xyz = atoms.xyz
        codes = []
        stereo: List[Tuple[int, float]] = []
        for n in range(first, last + 1):
            c = chain[n]
            hetero = [x for x in nb[c] if el[x] in HETEROATOMS]
            if n == a:
                codes.append("a")
                continue
            if n in (first, last):
                o_count = sum(1 for x in hetero if el[x] == "O")
                codes.append("A" if o_count >= 2 else ("h" if hetero else "m"))
                continue
            if not hetero:
                codes.append("d")
                continue
            if len(hetero) > 1:
                codes.append("?")
                residue.problem = f"C{n} has more than one heteroatom"
                continue
            v = _fischer(xyz, c, chain[n - 1], chain[n + 1], hetero[0])
            codes.append("2" if v < 0 else "1")
            if n > a:
                stereo.append((n, v))
        residue.skeleton = "".join(codes)
        # anomer: the exocyclic oxygen against the heteroatom of the reference atom; the ring bond of the
        # anomeric carbon is the "up" bond of the cyclic Fischer projection, for a ketose too (C1 then
        # stands beside it)
        up = ring_o
        down = chain.get(a + 1)
        if down is None or not stereo:
            residue.problem = residue.problem or "the anomeric reference atom cannot be found"
            continue
        residue.anomeric_volume = _fischer(xyz, anomeric, up, down, residue.exocyclic)
        reference_position, reference_volume = stereo[min(3, len(stereo) - 1)]
        residue.reference_position, residue.reference_volume = reference_position, reference_volume
        same_side = (residue.anomeric_volume < 0) == (reference_volume < 0)
        residue.anomer = "a" if same_side else "b"
    out.sort(key=lambda r: (r.key[0], r.key[1], r.key[2]))
    return out


def linkages(atoms: Atoms, residues: Sequence[SugarResidue], nb: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
    """Glycosidic linkages: every sugar whose exocyclic anomeric oxygen is bonded to a carbon of another sugar."""
    nb = neighbours(atoms) if nb is None else nb
    by_atom = {}
    for index, residue in enumerate(residues):
        for i in residue.atoms:
            by_atom[i] = index
    out = []
    for index, residue in enumerate(residues):
        if residue.exocyclic is None:
            continue
        o = residue.exocyclic
        for c in nb[o]:
            if c == residue.anomeric or atoms.elements[c] != "C":
                continue
            j = by_atom.get(c)
            if j is None or j == index:
                continue
            acceptor = residues[j]
            position = next((n for n, i in acceptor.chain.items() if i == c), None)
            out.append({"donor": index, "acceptor": j, "anomer": residue.anomer,
                        "donor_position": residue.anomeric_position, "acceptor_position": position})
    return out


@dataclass
class Reading:
    """What the atoms of one structure say: residues, linkages and the reducing-end residue."""

    residues: List[SugarResidue]
    linkages: List[Dict[str, Any]]
    reducing_end: Optional[int]

    def residue_codes(self) -> List[str]:
        return [r.code or "?" for r in self.residues]

    def linkage_keys(self) -> List[Tuple[str, str, int, Optional[int], str]]:
        """(donor skeleton and ring, anomer, donor position, acceptor position, acceptor skeleton and ring)."""
        out = []
        for link in self.linkages:
            d, a = self.residues[link["donor"]], self.residues[link["acceptor"]]
            out.append((_bare(d.code), link["anomer"] or "x", link["donor_position"], link["acceptor_position"],
                        _bare(a.code)))
        return out


def _bare(code: Optional[str]) -> str:
    """Skeleton and ring of a residue code, without the anomer: ``a2122h_1-5``."""
    if not code:
        return "?"
    head, _, ring = code.partition("_")
    return head.split("-")[0] + "_" + ring


def read(atoms: Atoms) -> Reading:
    nb = neighbours(atoms)
    residues = read_residues(atoms, nb)
    links = linkages(atoms, residues, nb)
    donors = {link["donor"] for link in links}
    roots = [i for i in range(len(residues)) if i not in donors]
    return Reading(residues, links, roots[0] if len(roots) == 1 else None)


# ---------------------------------------------------------------------------
# A WURCS string as the same residues and linkages
# ---------------------------------------------------------------------------
_WURCS = re.compile(r"^WURCS=2\.0/(\d+),(\d+),(\d+)/((?:\[[^\]]*\])+)/([^/]*)/(.*)$")


@dataclass
class Declared:
    residues: List[str]                    # codes (skeleton-anomeric position anomer_ring) of every residue
    linkages: List[Tuple[str, str, int, Optional[int], str]]
    reducing_end: int
    open_linkages: int = 0

    def residue_codes(self) -> List[str]:
        return list(self.residues)


def _wurcs_residue(code: str) -> Tuple[str, Optional[int], str]:
    """``a2122h-1b_1-5_2*NCC/3=O`` -> (``a2122h-1b_1-5``, anomeric position, anomer)."""
    parts = code.split("_")
    head = parts[0]
    ring = next((p for p in parts[1:] if re.match(r"^\d-\d$", p)), "")
    m = re.match(r"^([A-Za-z0-9]+?)(?:-(\d)([abx]))?$", head)
    if not m:
        return code, None, "x"
    skeleton, position, anomer = m.group(1), m.group(2), m.group(3) or "x"
    if position is None:                   # an open reducing end: u2122h -> a2122h, anomer unknown
        skeleton = "a" + skeleton[1:] if skeleton.startswith("u") else skeleton
        position = str(skeleton.index("a") + 1) if "a" in skeleton else None
    return f"{skeleton}-{position}{anomer}_{ring}", int(position) if position else None, anomer


def declared(wurcs: str) -> Declared:
    """Residues and linkages of a WURCS 2.0 string, in the terms of :class:`Reading`."""
    m = _WURCS.match(wurcs or "")
    if not m:
        raise ValueError(f"not WURCS 2.0: {str(wurcs)[:60]}")
    unique = [_wurcs_residue(c) for c in re.findall(r"\[([^\]]*)\]", m.group(4))]
    order = [int(i) - 1 for i in m.group(5).split("-")] if m.group(5) else []
    residues = [unique[u] for u in order]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = {letters[i]: i for i in range(len(residues))}
    links = []
    donors = set()
    open_links = 0
    for token in (m.group(6).split("_") if m.group(6) else []):
        if "~" in token or "|" in token or "?" in token or "-" not in token:
            open_links += 1
            continue
        left, right = token.split("-", 1)
        try:
            (i, pi), (j, pj) = [(index[s[0]], int(s[1:])) for s in (left, right)]
        except (KeyError, ValueError):
            open_links += 1
            continue
        # the donor is the side linked through its anomeric carbon (the later residue when both are)
        di = residues[i][1] == pi
        dj = residues[j][1] == pj
        if dj and (not di or j > i):
            donor, dp, acceptor, ap = j, pj, i, pi
        elif di:
            donor, dp, acceptor, ap = i, pi, j, pj
        else:
            open_links += 1
            continue
        donors.add(donor)
        links.append((_bare(residues[donor][0]), residues[donor][2], dp, ap, _bare(residues[acceptor][0])))
    roots = [i for i in range(len(residues)) if i not in donors]
    return Declared([r[0] for r in residues], links, roots[0] if roots else 0, open_links)


def compare(reading: Reading, name: Declared) -> Dict[str, Any]:
    """The structure against the name: ``match``, ``other_anomer`` (only the reducing-end anomer differs),
    ``different`` (with the residues and linkages that differ) or ``unread``.

    The outcomes are those of the GOTW name check as the thesis split them (``crabwurcs_data.py``,
    ``outcome``): ``other_anomer`` when the structure is the named glycan with the other anomer at the
    reducing end.
    """
    if not reading.residues or any(r.code is None for r in reading.residues):
        problems = [r.describe() for r in reading.residues if r.code is None]
        return {"outcome": "unread", "why": "no sugar residue found" if not reading.residues else
                "a residue could not be read", "residues": problems}
    alias = _wildcard_aliases(reading.residue_codes(), name.residue_codes())
    mine = Counter(alias.get(c, c) for c in reading.residue_codes())
    theirs = Counter(name.residue_codes())
    bare_alias = {_bare(k): _bare(v) for k, v in alias.items()}
    open_anomer = {_bare(v) for v in alias.values() if "-" in v and v.split("_")[0].endswith("x")}
    mine_links = Counter((bare_alias.get(d, d), "x" if bare_alias.get(d, d) in open_anomer else an, dp, ap,
                          bare_alias.get(a, a))
                         for d, an, dp, ap, a in reading.linkage_keys())
    their_links = Counter(name.linkages)
    result: Dict[str, Any] = {"outcome": "match"}
    if mine == theirs and (mine_links == their_links or name.open_linkages):
        return result
    # the reducing end alone: the same molecule with the other anomer at the reducing end
    if reading.reducing_end is not None:
        root = reading.residues[reading.reducing_end]
        root_code = alias.get(root.code, root.code)
        flipped = Counter(mine)
        other = {"a": "b", "b": "a"}.get(root.anomer or "")
        if other:
            flipped[root_code] -= 1
            flipped[root_code.replace(f"-{root.anomeric_position}{root.anomer}_", f"-{root.anomeric_position}{other}_")] += 1
            flipped = +flipped
            if flipped == theirs and (mine_links == their_links or name.open_linkages):
                return {"outcome": "other_anomer", "structure_reducing_end": root.describe(),
                        "named_reducing_end_anomer": other}
    residue_only_mine = sorted((mine - theirs).elements())
    residue_only_name = sorted((theirs - mine).elements())
    link_only_mine = [list(x) for x in sorted((mine_links - their_links).elements(), key=str)]
    link_only_name = [list(x) for x in sorted((their_links - mine_links).elements(), key=str)]
    root_anomers = None
    if reading.reducing_end is not None and 0 <= name.reducing_end < len(name.residues):
        named_root = name.residues[name.reducing_end]
        root_anomers = {"structure": reading.residues[reading.reducing_end].anomer,
                        "name": named_root.split("_")[0][-1:] or None}
    return {"outcome": "different", "reducing_end_anomer": root_anomers,
            "residues_in_structure_not_in_name": residue_only_mine,
            "residues_in_name_not_in_structure": residue_only_name,
            "linkages_in_structure_not_in_name": link_only_mine,
            "linkages_in_name_not_in_structure": link_only_name,
            "kinds": difference_kinds(residue_only_mine, residue_only_name, link_only_mine, link_only_name)}


def _code_fits(code: str, pattern: str) -> bool:
    """Whether a residue code read from atoms fits a declared code with ``x`` (unknown) characters or no ring."""
    head, _, ring = code.partition("_")
    p_head, _, p_ring = pattern.partition("_")
    if p_ring and ring != p_ring:
        return False
    return len(head) == len(p_head) and all(b == "x" or a == b for a, b in zip(head, p_head))


def _wildcard_aliases(structure: Sequence[str], names: Sequence[str]) -> Dict[str, str]:
    """Structure codes to read as a declared code that leaves characters unknown (``axxxxm-1x_``)."""
    patterns = [c for c in set(names) if "x" in c.partition("_")[0] or not c.partition("_")[2]]
    out: Dict[str, str] = {}
    for code in set(structure):
        if code in names:
            continue
        fits = [p for p in patterns if _code_fits(code, p)]
        if len(fits) == 1:
            out[code] = fits[0]
    return out


def difference_kinds(res_mine: List[str], res_name: List[str], link_mine: List[list], link_name: List[list]) -> List[str]:
    """Plain words for a difference: ``anomer``, ``configuration``, ``ring``, ``residue``, ``linkage position``."""
    kinds = set()
    for a, b in zip(sorted(res_mine), sorted(res_name)):
        sa, sb = a.split("-")[0], b.split("-")[0]
        ra, rb = a.split("_", 1)[-1], b.split("_", 1)[-1]
        if sa == sb and ra == rb:
            kinds.add("anomer")
        elif sa == sb:
            kinds.add("ring")
        elif len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) and \
                all(x == y or {x, y} == {"1", "2"} for x, y in zip(sa, sb)):
            kinds.add("configuration")
        else:
            kinds.add("residue")
    if len(res_mine) != len(res_name):
        kinds.add("residue count")
    if not res_mine and not res_name and (link_mine or link_name):
        for a, b in zip(link_mine, link_name):
            if a[1] != b[1]:
                kinds.add("linkage anomer")
            if a[2] != b[2] or a[3] != b[3]:
                kinds.add("linkage position")
            if a[0] != b[0] or a[4] != b[4]:
                kinds.add("linkage residues")
        if len(link_mine) != len(link_name):
            kinds.add("linkage count")
    return sorted(kinds)


# ---------------------------------------------------------------------------
# Stereocentres (the configuration test of the thesis' stereocentre analysis)
# ---------------------------------------------------------------------------
def stereocentres(atoms: Atoms, nb: Optional[List[List[int]]] = None) -> List[Tuple[int, List[int]]]:
    """Carbons with four bonded atoms of which at least three are not hydrogen: (carbon, three heavy neighbours).

    Port of ``stereocentres`` of ``robustness_data.py``: bonds from the distances of one frame; the three
    neighbours are the first three in the order atoms of the carbon's own residue before atoms of another
    residue, then by atom name.
    """
    nb = neighbours(atoms) if nb is None else nb
    el = atoms.elements
    out = []
    for i in range(len(atoms)):
        if el[i] != "C":
            continue
        heavy = [j for j in nb[i] if el[j] != "H"]
        if len(nb[i]) == 4 and len(heavy) >= 3:
            heavy = sorted(heavy, key=lambda j: (atoms.residue_keys[j] != atoms.residue_keys[i], atoms.names[j], j))[:3]
            out.append((i, heavy))
    return out


def centre_signs(xyz: np.ndarray, centres: Sequence[Tuple[int, List[int]]]) -> np.ndarray:
    """Whether the signed volume of each centre is positive, for frames ``xyz`` [frames, atoms, 3]."""
    if not centres:
        return np.zeros((len(xyz), 0), dtype=bool)
    ci = np.array([c for c, _ in centres], dtype=int)
    cn = np.array([n for _, n in centres], dtype=int).reshape(-1, 3)
    x = np.asarray(xyz, dtype=np.float64)
    o = x[:, ci, :]
    v = np.einsum("fci,fci->fc", x[:, cn[:, 0], :] - o, np.cross(x[:, cn[:, 1], :] - o, x[:, cn[:, 2], :] - o))
    return v > 0


def centre_label(atoms: Atoms, centre: Tuple[int, List[int]]) -> str:
    """``0LB C1 (C2, O5, O4)``: the key of a centre type in the thesis (residue name, atom, neighbours)."""
    i, nb3 = centre
    return f"{atoms.residue_names[i][:3]} {atoms.names[i]} ({', '.join(atoms.names[j] for j in nb3)})"
