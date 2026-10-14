"""Glycan classes of the search keywords, decided on the residue tree.

A glycan is read as a tree of residues rooted at its reducing end, from its
WURCS (the identity record GlyTouCan registers) or from its IUPAC-condensed
name, and the class rules are tests on that tree: which residue is at the
root, which residues hang from it and through which linkage.  Sulfate,
methyl, acetyl and phosphocholine groups are ignored when a residue is
compared with a residue name (``Residue.base``: ``GlcNS6S`` is ``GlcNS``,
``Neu5Ac9Ac`` is ``Neu5Ac``).

The rules are those of the rule-based classification of the GlycoShape
collection in Ojas Singh's thesis (``figures/GlycoShape/scripts/coverage_classes.py``),
restricted to the classes that the keywords name.  They are tried in the
thesis's order, so a glycan has at most one of the three classes:

1. **N-glycan**: the root is GlcNAc, it carries GlcNAc(b1-4), and that carries
   Man(b1-4).  Type from the two alpha-Man arms on the beta-Man: *complex*
   when an arm carries GlcNAc and no arm carries a further Man; *hybrid* when
   an arm carries GlcNAc and an arm carries a further Man; *oligomannose* when
   the arms carry only Man; *core* when neither arm carries anything (at most
   Man3GlcNAc2, with or without core Fuc or Xyl).  The antennae of a complex
   N-glycan are the GlcNAc residues on the alpha-Man arms (a bisecting GlcNAc
   on the beta-Man is not one).  Core fucose: Fuc(a1-6) or Fuc(a1-3) on the
   root GlcNAc, whatever the Fuc carries.
2. **Glycosaminoglycan**: at least one bond of a glycosaminoglycan repeat
   (``GAG_BONDS``) and nothing but uronic acids and hexosamines outside the
   linkage region, or the linkage region GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl at
   the root.
3. **O-GalNAc glycan** (mucin type): the root is GalNAc, substituted only at
   O-3 and O-6 with one of Brockhausen's cores 1 to 8 or sialyl-Tn, and no
   residue that vertebrate glycoconjugates do not use (such a glycan is
   microbial or plant in the thesis, whose rule comes first).  Ganglio- and
   globo-series headgroups have Glc at the root and are not O-glycans.

A monosaccharide has none of the classes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SIALIC = {"Neu5Ac", "Neu5Gc", "Kdn"}
URONIC = {"GlcA", "IdoA"}
HEXOSAMINE = {"GlcNAc", "GlcNS", "GalNAc"}
# Residues of vertebrate glycoconjugates; any other residue marks a glycan as microbial or plant.
VERTEBRATE = {"Glc", "Gal", "Man", "GlcNAc", "GalNAc", "Fuc", "Xyl", "GlcA", "IdoA", "GlcNS"} | SIALIC

# (stem, absolute configuration that is written without a prefix)
COMMON_CONFIG = {"Glc": "D", "Gal": "D", "Man": "D", "All": "D", "Alt": "D", "Gul": "D", "Tal": "D", "Ido": "L",
                 "Fuc": "L", "Rha": "L", "Qui": "D", "Xyl": "D", "Ara": "L", "Lyx": "D", "Rib": "D", "Fru": "D",
                 "Psi": "D", "Tag": "D", "Sor": "D", "Neu": "D", "Kdn": "D", "Kdo": "D", "Bac": "D", "dHex": "?"}


@dataclass
class Residue:
    name: str                    # IUPAC-condensed style, e.g. GlcNAc, IdoA2S, D-Araf
    base: str                    # the same without sulfate, methyl, acetyl or phosphocholine groups
    stem: str                    # Glc, Gal, Neu ...
    config: str                  # D, L or ?
    ring: str                    # p, f or ? (WURCS does not state the ring of an open reducing end)
    anomer: str                  # a, b or ?
    anomeric_pos: int            # 1, or 2 for ketoses and sialic acids
    parent: Optional[int] = None
    parent_pos: Optional[str] = None  # acceptor position on the parent ("3", or "3/6" where it is left open)
    children: List[int] = field(default_factory=list)


def _make_name(stem: str, config: str, ring: str, core_mod: str, subs: List[Tuple[int, str]]) -> Tuple[str, str]:
    """(name, base): Glc + NAc + 6S -> (GlcNAc6S, GlcNAc); Ara, furanose, D -> (D-Araf, D-Araf)."""
    prefix = "" if config in (COMMON_CONFIG.get(stem), "?") else f"{config}-"
    base = f"{prefix}{stem}{'f' if ring == 'f' else ''}{core_mod}"
    return base + "".join(f"{p}{s}" for p, s in sorted(subs)), base


# --- WURCS 2.0 --------------------------------------------------------------------------------------------------
ALDOHEXOSE = {"2122": ("Glc", "D"), "2112": ("Gal", "D"), "1122": ("Man", "D"), "2222": ("All", "D"),
              "2111": ("Alt", "L"), "1121": ("Gul", "L"), "2221": ("Tal", "L"), "2211": ("Man", "L"),
              "2121": ("Ido", "L")}
DEOXYHEXOSE = {"1221": ("Fuc", "L"), "2112": ("Fuc", "D"), "2211": ("Rha", "L"), "1122": ("Rha", "D"),
               "2122": ("Qui", "D"), "xxxx": ("dHex", "?")}
PENTOSE = {"212": ("Xyl", "D"), "211": ("Ara", "L"), "122": ("Ara", "D"), "112": ("Lyx", "D"), "222": ("Rib", "D")}
KETOHEXOSE = {"122": ("Fru", "D"), "222": ("Psi", "D"), "112": ("Tag", "D"), "121": ("Sor", "L")}
WURCS_SUBSTITUENT = {"OSO/3=O/3=O": "S", "OC": "Me", "OCC/3=O": "Ac", "OP^XOCCNC/7C/7C/3O/3=O": "PCho"}


def _decode_wurcs_residue(code: str) -> Dict:
    """One unique-residue code of WURCS 2.0 -> stem, configuration, ring, anomer, substituents."""
    parts = code.split("_")
    head, mods = parts[0], parts[1:]
    m = re.match(r"^([A-Za-z0-9]+?)(?:-(\d)([abx]))?$", head)
    if not m:
        raise ValueError(f"WURCS residue head not understood: {code}")
    skeleton, apos, anomer = m.group(1), m.group(2), m.group(3)
    ring = "?"
    subs: List[Tuple[int, str]] = []
    core_mod = ""
    for mod in mods:
        r = re.match(r"^(\d)-(\d)$", mod)
        if r:
            ring = {4: "p", 3: "f"}.get(int(r.group(2)) - int(r.group(1)), "?")
            continue
        s = re.match(r"^(\d)\*(.+)$", mod)
        if not s:
            raise ValueError(f"WURCS modification not understood: {mod} in {code}")
        pos, what = int(s.group(1)), s.group(2)
        if what == "NCC/3=O":
            subs.append((pos, "NAc"))
        elif what == "NCCO/3=O":
            subs.append((pos, "NGc"))
        elif what == "NSO/3=O/3=O":
            subs.append((pos, "NS"))
        elif what in WURCS_SUBSTITUENT:
            subs.append((pos, WURCS_SUBSTITUENT[what]))
        else:
            raise ValueError(f"WURCS substituent not known: {what} in {code}")

    if skeleton[:2] in ("Aa", "AU"):                       # 2-keto-3-deoxy acids: Neu, Kdn, Kdo
        body = skeleton[3:-1] if len(skeleton) > 3 and skeleton[2] == "d" else None
        if body == "21122":
            n5 = [s for s in subs if s[0] == 5 and s[1] in ("NAc", "NGc")]
            subs = [s for s in subs if s not in n5]
            if n5:
                stem, core_mod = "Neu", {"NAc": "5Ac", "NGc": "5Gc"}[n5[0][1]]
            else:
                stem = "Kdn"
        elif body == "1122":
            stem = "Kdo"
        else:
            raise ValueError(f"WURCS skeleton not known: {code}")
        config, apos = "D", apos or "2"
        if ring == "?":
            ring = "p"
    elif skeleton[0] == "h":                               # ketohexoses
        if skeleton[2:-1] not in KETOHEXOSE:
            raise ValueError(f"WURCS skeleton not known: {code}")
        stem, config = KETOHEXOSE[skeleton[2:-1]]
        apos = apos or "2"
    else:
        body, last = skeleton[1:-1], skeleton[-1]
        table = (ALDOHEXOSE if len(body) == 4 and last in "hA" else DEOXYHEXOSE if len(body) == 4 and last == "m"
                 else PENTOSE if len(body) == 3 and last == "h" else {})
        if body not in table:
            raise ValueError(f"WURCS skeleton not known: {code}")
        stem, config = table[body]
        if last == "A":
            core_mod = "A"
        apos = apos or "1"
        n2 = [s for s in subs if s[0] == 2 and s[1] in ("NAc", "NS")]
        if n2:
            n4 = [s for s in subs if s[0] == 4 and s[1] == "NAc"]
            if n4 and last == "m":                         # 2,4-diacetamido-2,4,6-trideoxyhexose
                subs = [s for s in subs if s not in n2 + n4]
                stem = "Bac" if stem == "Qui" else stem
                core_mod = "2Ac4Ac" if stem == "Bac" else "2NAc4NAc"
            else:
                subs = [s for s in subs if s not in n2]
                core_mod = n2[0][1] + core_mod
    return {"stem": stem, "config": config, "ring": ring, "anomer": {"a": "a", "b": "b"}.get(anomer, "?"),
            "anomeric_pos": int(apos), "core_mod": core_mod, "subs": subs}


def parse_wurcs(wurcs: str) -> List[Residue]:
    """Residues of a WURCS 2.0 string as a tree rooted at its first residue (the reducing end)."""
    m = re.match(r"^WURCS=2\.0/(\d+),(\d+),(\d+)/((?:\[[^\]]*\])+)/([^/]*)/(.*)$", str(wurcs or "").strip())
    if not m:
        raise ValueError(f"not WURCS 2.0: {str(wurcs)[:60]}")
    n_res = int(m.group(2))
    unique = [_decode_wurcs_residue(c) for c in re.findall(r"\[([^\]]*)\]", m.group(4))]
    order = [int(i) - 1 for i in m.group(5).split("-")]
    if len(order) != n_res or n_res > 52:
        raise ValueError("residue count of the WURCS header and its sequence disagree")
    residues = []
    for u in order:
        d = unique[u]
        name, base = _make_name(d["stem"], d["config"], d["ring"], d["core_mod"], d["subs"])
        residues.append(Residue(name, base, d["stem"], d["config"], d["ring"], d["anomer"], d["anomeric_pos"]))
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = {letters[i]: i for i in range(n_res)}
    adjacency: Dict[int, list] = {i: [] for i in range(n_res)}
    for token in (m.group(6).split("_") if m.group(6) else []):
        if not token:
            continue
        if token.count("-") != 1:
            raise ValueError(f"WURCS linkage not understood: {token}")
        ends = []
        for side in token.split("-"):
            alts = side.split("|")
            found = {a[:1] for a in alts}
            if len(found) != 1 or next(iter(found)) not in index:
                raise ValueError(f"WURCS linkage with alternative residues or not understood: {token}")
            ends.append((index[found.pop()], "/".join(a[1:] for a in alts)))
        (i, pi), (j, pj) = ends
        adjacency[i].append((j, pi, pj))
        adjacency[j].append((i, pj, pi))
    seen, queue = {0}, [0]
    while queue:
        parent = queue.pop(0)
        for child, parent_pos, child_pos in adjacency[parent]:
            if child in seen:
                continue
            seen.add(child)
            residues[child].parent, residues[child].parent_pos = parent, parent_pos
            if child_pos != str(residues[child].anomeric_pos):
                raise ValueError(f"residue {child} is linked through position {child_pos}, not its anomeric carbon")
            residues[parent].children.append(child)
            queue.append(child)
    if len(seen) != n_res:
        raise ValueError("WURCS linkages do not connect all residues")
    return residues


# --- IUPAC condensed --------------------------------------------------------------------------------------------
_IUPAC_RESIDUE = re.compile(
    r"^(?:(?P<config>[DL])-)?"
    r"(?P<stem>Glc|Gal|Man|Fuc|Neu|Xyl|Rha|Ara|Fru|Ido|Kdn|KDN|Kdo|KDO|Qui|Rib|Lyx|All|Alt|Gul|Tal|Psi|Tag|Sor|Bac|dHex)"
    r"(?P<ring>[pf])?(?P<mod>NAc|NGc|NS|A|5Ac|5Gc)?(?P<subs>(?:\d(?:S|Me|Ac|PCho))*)$")
_IUPAC_LINK = re.compile(r"\((?P<anomer>[ab?])(?P<apos>\d)-(?P<to>[0-9?]+(?:/[0-9?]+)*)\)")
_IUPAC_NAME = re.compile(r"[^()\[\]]+")
_ANOMERIC_C2 = {"Neu", "Kdn", "Kdo", "Fru", "Psi", "Tag", "Sor"}


def _iupac_residue(text: str, anomer: str, apos: Optional[int]) -> Residue:
    m = _IUPAC_RESIDUE.match(text)
    if not m:
        raise ValueError(f"IUPAC residue not understood: {text}")
    stem = {"KDN": "Kdn", "KDO": "Kdo"}.get(m.group("stem"), m.group("stem"))
    config = m.group("config") or COMMON_CONFIG.get(stem, "D")
    ring = m.group("ring") or "p"
    subs = [(int(s[0]), s[1:]) for s in re.findall(r"\d(?:PCho|Me|Ac|S)", m.group("subs") or "")]
    name, base = _make_name(stem, config, ring, m.group("mod") or "", subs)
    if apos is None:                                       # the reducing end: no linkage is written
        apos = 2 if stem in _ANOMERIC_C2 else 1
    return Residue(name, base, stem, config, ring, anomer, apos)


def parse_iupac_condensed(text: str) -> List[Residue]:
    """An IUPAC-condensed name (``Gal(b1-4)[Fuc(a1-3)]GlcNAc``) as a tree; the last residue is the root."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty IUPAC name")
    residues: List[Residue] = []
    pending: List[Tuple[int, str]] = []
    stack: List[List[Tuple[int, str]]] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "[":
            stack.append(pending)
            pending = []
            pos += 1
            continue
        if ch == "]":
            if not stack:
                raise ValueError(f"unbalanced bracket at {pos}: {text[:60]}")
            pending = stack.pop() + pending
            pos += 1
            continue
        m = _IUPAC_NAME.match(text, pos)
        if not m:
            raise ValueError(f"IUPAC name not understood at {pos}: {text[pos:pos + 30]}")
        pos = m.end()
        link = _IUPAC_LINK.match(text, pos)
        if link:
            residue = _iupac_residue(m.group(0), link.group("anomer"), int(link.group("apos")))
            pos = link.end()
        elif pos == len(text):
            residue = _iupac_residue(m.group(0), "?", None)
        else:
            raise ValueError(f"IUPAC linkage not understood at {pos}: {text[pos:pos + 30]}")
        residues.append(residue)
        me = len(residues) - 1
        for child, to in pending:
            residues[child].parent, residues[child].parent_pos = me, to
            residue.children.append(child)
        pending = [] if not link else [(me, link.group("to"))]
    if stack or pending:
        raise ValueError(f"IUPAC name does not end at a reducing end: {text[:60]}")
    return residues


# --- the rules --------------------------------------------------------------------------------------------------
def _sulfate_groups(x: Residue) -> int:
    """O-sulfate groups written after the residue name, plus the N-sulfate of GlcNS."""
    return len(re.findall(r"\dS", x.name[len(x.base):])) + (1 if "NS" in x.base else 0)


class Tree:
    def __init__(self, residues: List[Residue]):
        self.r = residues
        self.root = next(i for i, x in enumerate(residues) if x.parent is None)
        self.n = len(residues)
        self.bases = [x.base for x in residues]

    def link(self, c: int) -> str:
        x = self.r[c]
        return f"{x.anomer}{x.anomeric_pos}-{x.parent_pos}"

    def kids(self, i: int, base=None, link: Optional[str] = None) -> List[int]:
        """Children of residue i, optionally with a given base name (or set of names) and linkage ('b1-4')."""
        out = []
        for c in self.r[i].children:
            if base is not None and self.r[c].base not in ({base} if isinstance(base, str) else base):
                continue
            if link is not None and self.link(c) != link:
                continue
            out.append(c)
        return out

    def bonds(self) -> List[Tuple[str, str, str]]:
        """(donor base, linkage, acceptor base) of every glycosidic bond."""
        return [(x.base, self.link(i), self.r[x.parent].base) for i, x in enumerate(self.r) if x.parent is not None]

    def subtree(self, i: int) -> List[int]:
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(self.r[j].children)
        return out


def n_glycan(t: Tree) -> Optional[Dict]:
    """Root GlcNAc <- GlcNAc(b1-4) <- Man(b1-4): the type of the N-glycan, or None."""
    if t.bases[t.root] != "GlcNAc":
        return None
    for g2 in t.kids(t.root, "GlcNAc", "b1-4"):
        for bman in t.kids(g2, "Man", "b1-4"):
            arms = [c for c in t.kids(bman, "Man") if t.r[c].anomer == "a"]       # a1-3 and a1-6 (or left open)
            n_ant = sum(len(t.kids(a, "GlcNAc")) for a in arms)
            n_arm_man = sum(len(t.kids(a, "Man")) for a in arms)
            if n_ant and n_arm_man:
                kind = "hybrid"
            elif n_ant:
                kind = "complex"
            elif n_arm_man:
                kind = "oligomannose"
            else:
                kind = "core"                                   # at most the trimannosyl core
            return {
                "kind": kind,
                "antennae": n_ant,
                "core_fuc6": bool(t.kids(t.root, "Fuc", "a1-6")),
                "core_fuc3": bool(t.kids(t.root, "Fuc", "a1-3")),
            }
    return None


GAG_BONDS = {
    "heparan sulfate / heparin": {("GlcA", "b1-4", "GlcNAc"), ("GlcA", "b1-4", "GlcNS"), ("IdoA", "a1-4", "GlcNAc"),
                                  ("IdoA", "a1-4", "GlcNS"), ("GlcNAc", "a1-4", "GlcA"), ("GlcNAc", "a1-4", "IdoA"),
                                  ("GlcNS", "a1-4", "GlcA"), ("GlcNS", "a1-4", "IdoA")},
    "chondroitin / dermatan sulfate": {("GlcA", "b1-3", "GalNAc"), ("IdoA", "a1-3", "GalNAc"),
                                       ("GalNAc", "b1-4", "GlcA"), ("GalNAc", "b1-4", "IdoA")},
    "hyaluronan": {("GlcA", "b1-3", "GlcNAc"), ("GlcNAc", "b1-4", "GlcA")},
}


def _linkage_region(t: Tree) -> Optional[List[int]]:
    """GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl at the reducing end: the residues of the tetrasaccharide, or None."""
    if t.bases[t.root] != "Xyl":
        return None
    for g1 in t.kids(t.root, "Gal", "b1-4"):
        for g2 in t.kids(g1, "Gal", "b1-3"):
            for u in t.kids(g2, "GlcA", "b1-3"):
                return [t.root, g1, g2, u]
    return None


def gag(t: Tree) -> Optional[Dict]:
    region = _linkage_region(t)
    bonds = t.bonds()
    found = {name: sum(1 for b in bonds if b in kinds) for name, kinds in GAG_BONDS.items()}
    found = {k: v for k, v in found.items() if v}
    rest = [i for i in range(t.n) if i not in (region or [])]
    only_gag_residues = all(t.bases[i] in URONIC | HEXOSAMINE for i in rest)
    if region and not found:
        return {"kind": "linkage region"}
    if found and (only_gag_residues or region):
        return {"kind": max(found, key=found.get), "sulfated": any(_sulfate_groups(x) for x in t.r)}
    return None


def foreign_residues(t: Tree) -> List[str]:
    """Residues that vertebrate glycoconjugates do not use, including methylated and acetylated neutral sugars."""
    out = set()
    for x in t.r:
        tail = x.name[len(x.base):]
        if x.base not in VERTEBRATE or x.config == "?" or x.ring == "f":
            out.add(x.name)
        elif x.base not in SIALIC and ("Me" in tail or "Ac" in tail or "PCho" in tail):
            out.add(x.name)
    return sorted(out)


def o_galnac(t: Tree) -> Optional[Dict]:
    """GalNAc at the reducing end; the core from what it carries at O-3 and O-6 (Brockhausen's cores 1 to 8)."""
    if t.bases[t.root] != "GalNAc" or t.n < 2:
        return None
    on: Dict[Optional[str], list] = {}
    for c in t.r[t.root].children:
        on.setdefault(t.r[c].parent_pos, []).append((t.bases[c], t.r[c].anomer))
    if set(on) - {"3", "6"} or any(len(v) != 1 for v in on.values()):
        return None
    at3, at6 = (on.get("3") or [None])[0], (on.get("6") or [None])[0]
    six_sia = at6 is not None and at6[0] in SIALIC
    if at3 == ("Gal", "b") and at6 == ("GlcNAc", "b"):
        core = "core 2"
    elif at3 == ("Gal", "b") and (at6 is None or six_sia):
        core = "core 1"
    elif at3 == ("GlcNAc", "b") and at6 == ("GlcNAc", "b"):
        core = "core 4"
    elif at3 == ("GlcNAc", "b") and (at6 is None or six_sia):
        core = "core 3"
    elif at3 == ("GalNAc", "a") and (at6 is None or six_sia):
        core = "core 5"
    elif at3 is None and at6 == ("GlcNAc", "b"):
        core = "core 6"
    elif at3 is None and at6 == ("GalNAc", "a"):
        core = "core 7"
    elif at3 == ("Gal", "a") and at6 is None:
        core = "core 8"
    elif at3 is None and six_sia:
        core = "sialyl-Tn"
    else:
        return None
    return {"kind": core}


def classify_tree(residues: List[Residue]) -> Optional[Tuple[str, Dict]]:
    """``("n_glycan" | "gag" | "o_galnac", detail)`` of a residue tree, or None."""
    t = Tree(residues)
    if t.n < 2:
        return None
    for key, rule in (("n_glycan", n_glycan), ("gag", gag)):
        detail = rule(t)
        if detail:
            return key, detail
    if foreign_residues(t):
        return None
    detail = o_galnac(t)
    return ("o_galnac", detail) if detail else None


def residue_tree(wurcs: Optional[str] = None, iupac: Optional[str] = None) -> Tuple[Optional[List[Residue]], str]:
    """The residue tree of a glycan and where it was read from (``"wurcs"``, ``"iupac"``, or why there is none).

    The WURCS is read first; the IUPAC-condensed name when there is no WURCS or it cannot be read.
    """
    problems = []
    for source, text, parse in (("wurcs", wurcs, parse_wurcs), ("iupac", iupac, parse_iupac_condensed)):
        if not text or str(text).strip().lower() in ("", "none", "null", "nan"):
            continue
        try:
            return parse(str(text)), source
        except (ValueError, KeyError, IndexError, AttributeError) as exc:
            problems.append(f"{source}: {exc}")
    return None, "; ".join(problems) or "neither WURCS nor IUPAC given"


CLASS_KEYWORDS = (
    "N-Glycan", "Fucosylated N-Glycan", "Oligomannose", "Complex N-Glycan", "Hybrid N-Glycan",
    "Biantennary N-Glycan", "Triantennary N-Glycan", "Tetraantennary N-Glycan", "O-Glycan", "GAG",
)
ANTENNAE_KEYWORD = {2: "Biantennary N-Glycan", 3: "Triantennary N-Glycan", 4: "Tetraantennary N-Glycan"}


def keywords_of_class(hit: Optional[Tuple[str, Dict]]) -> List[str]:
    """The class keywords of a classification (:func:`classify_tree`), sorted.

    ``Fucosylated N-Glycan`` means a core fucose (a1-6 or a1-3 on the root GlcNAc).
    The antennary keywords are given to complex N-glycans with two, three or four
    antennae; a mono- or pentaantennary one has no antennary keyword.
    """
    if not hit:
        return []
    key, detail = hit
    if key == "gag":
        return ["GAG"]
    if key == "o_galnac":
        return ["O-Glycan"]
    keywords = {"N-Glycan"}
    if detail["core_fuc6"] or detail["core_fuc3"]:
        keywords.add("Fucosylated N-Glycan")
    if detail["kind"] == "oligomannose":
        keywords.add("Oligomannose")
    elif detail["kind"] == "hybrid":
        keywords.add("Hybrid N-Glycan")
    elif detail["kind"] == "complex":
        keywords.add("Complex N-Glycan")
        if detail["antennae"] in ANTENNAE_KEYWORD:
            keywords.add(ANTENNAE_KEYWORD[detail["antennae"]])
    return sorted(keywords)


def class_keywords(wurcs: Optional[str] = None, iupac: Optional[str] = None) -> Tuple[List[str], str]:
    """Class keywords of a glycan from its WURCS or IUPAC-condensed name, and the source of its tree."""
    tree, source = residue_tree(wurcs, iupac)
    if tree is None:
        return [], source
    return keywords_of_class(classify_tree(tree)), source
