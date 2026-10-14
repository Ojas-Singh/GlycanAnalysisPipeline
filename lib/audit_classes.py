"""Glycan classes from the residue tree, for the entry audit (``lib/audit.py``).

A port of the class rules of the coverage figure of the thesis
(``figures/GlycoShape/scripts/coverage_glycan.py`` and ``coverage_classes.py``
of the thesis repository; their output is ``data/coverage-classes.json``),
kept as they are so that the audit flags the entries the thesis flagged:

* :func:`parse_wurcs` and :func:`parse_glycam` read the two records of an
  entry (its WURCS and its GLYCAM name) as trees rooted at the reducing end,
  with the small residue table of the thesis;
* :func:`classify` gives the class of a tree (N-glycan, O-GalNAc,
  glycosaminoglycan, ...), the other classes whose rule also holds, and the
  details from which the audit reads what is not canonical: an N-glycan-like
  glycan whose core is not Man(b1-4)GlcNAc(b1-4)GlcNAc (``n_glycan_like``),
  an unusual substituent on the beta-Man of the core
  (``unusual_on_beta_man``) and linkage positions left open
  (``open_linkages``).

Two independent records of each entry are parsed: its WURCS string (the
identity record GlyTouCan registers) and its GLYCAM sequence name (the name
under which it was built and simulated).  The WURCS residue codes are decoded
with the table below, written for the residues of the collection; a code
outside it raises ``ValueError``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- residue vocabulary ---------------------------------------------------------------------------------------
# (stem, absolute configuration that is written without a prefix)
COMMON_CONFIG = {"Glc": "D", "Gal": "D", "Man": "D", "All": "D", "Alt": "D", "Gul": "D", "Tal": "D", "Ido": "L",
                 "Fuc": "L", "Rha": "L", "Qui": "D", "Xyl": "D", "Ara": "L", "Lyx": "D", "Rib": "D", "Fru": "D",
                 "Psi": "D", "Tag": "D", "Sor": "D", "Neu": "D", "Kdn": "D", "Kdo": "D", "Bac": "D", "dHex": "?"}

# WURCS skeleton codes: stereo descriptors of the carbons between the anomeric carbon and the last carbon
ALDOHEXOSE = {"2122": ("Glc", "D"), "2112": ("Gal", "D"), "1122": ("Man", "D"), "2222": ("All", "D"),
              "2111": ("Alt", "L"), "1121": ("Gul", "L"), "2221": ("Tal", "L"), "2211": ("Man", "L"),
              "2121": ("Ido", "L")}
DEOXYHEXOSE = {"1221": ("Fuc", "L"), "2112": ("Fuc", "D"), "2211": ("Rha", "L"), "1122": ("Rha", "D"),
               "2122": ("Qui", "D"), "xxxx": ("dHex", "?")}
PENTOSE = {"212": ("Xyl", "D"), "211": ("Ara", "L"), "122": ("Ara", "D"), "112": ("Lyx", "D"), "222": ("Rib", "D")}
KETOHEXOSE = {"122": ("Fru", "D"), "222": ("Psi", "D"), "112": ("Tag", "D"), "121": ("Sor", "L")}

SUBSTITUENT = {"OSO/3=O/3=O": "S", "OC": "Me", "OCC/3=O": "Ac", "OP^XOCCNC/7C/7C/3O/3=O": "PCho"}


@dataclass
class Residue:
    name: str                    # IUPAC-condensed style, e.g. GlcNAc, IdoA2S, D-Araf
    base: str                    # the same without sulfate, methyl, acetyl or phosphocholine groups (IdoA2S -> IdoA)
    stem: str                    # Glc, Gal, Neu ...
    config: str                  # D, L or ?
    ring: str                    # p, f or ? (not given for an open reducing end in WURCS)
    anomer: str                  # a, b or ? (reducing end without a stated anomer)
    anomeric_pos: int            # 1, or 2 for ketoses and sialic acids
    parent: int | None = None
    parent_pos: str | None = None  # acceptor position on the parent ("3", or "3/6" where the record leaves it open)
    children: list = field(default_factory=list)


def make_name(stem: str, config: str, ring: str, core_mod: str, subs: list[tuple[int, str]]) -> tuple[str, str]:
    """(name, base): Glc + NAc + 6S -> (GlcNAc6S, GlcNAc); Ara, furanose, D -> (D-Araf, D-Araf)."""
    prefix = "" if config in (COMMON_CONFIG.get(stem), "?") else f"{config}-"
    base = f"{prefix}{stem}{'f' if ring == 'f' else ''}{core_mod}"
    return base + "".join(f"{p}{s}" for p, s in sorted(subs)), base


# --- WURCS ----------------------------------------------------------------------------------------------------
def decode_wurcs_residue(code: str) -> dict:
    """One unique-residue code of WURCS 2.0 -> stem, configuration, ring, anomer, substituents."""
    parts = code.split("_")
    head = parts[0]
    mods = parts[1:]
    m = re.match(r"^([A-Za-z0-9]+?)(?:-(\d)([abx]))?$", head)
    if not m:
        raise ValueError(f"WURCS residue head not understood: {code}")
    skeleton, apos, anomer = m.group(1), m.group(2), m.group(3)
    ring = "?"
    subs: list[tuple[int, str]] = []
    core_mod = ""
    for mod in mods:
        r = re.match(r"^(\d)-(\d)$", mod)
        if r:
            span = int(r.group(2)) - int(r.group(1))
            ring = {4: "p", 3: "f"}.get(span, "?")
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
        elif what in SUBSTITUENT:
            subs.append((pos, SUBSTITUENT[what]))
        else:
            raise ValueError(f"WURCS substituent not in the table: {what} in {code}")

    if skeleton[:2] in ("Aa", "AU"):                       # 2-keto-3-deoxy acids: Neu, Kdn, Kdo
        body = skeleton[3:-1] if skeleton[2] == "d" else None
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
            raise ValueError(f"WURCS skeleton not in the table: {code}")
        config, apos = "D", apos or "2"
        if ring == "?":
            ring = "p"
    elif skeleton[0] == "h":                               # ketohexoses
        stem, config = KETOHEXOSE[skeleton[2:-1]]
        apos = apos or "2"
    else:
        body, last = skeleton[1:-1], skeleton[-1]
        if len(body) == 4 and last == "h":
            stem, config = ALDOHEXOSE[body]
        elif len(body) == 4 and last == "m":
            stem, config = DEOXYHEXOSE[body]
        elif len(body) == 4 and last == "A":
            stem, config = ALDOHEXOSE[body]
            core_mod = "A"
        elif len(body) == 3 and last == "h":
            stem, config = PENTOSE[body]
        else:
            raise ValueError(f"WURCS skeleton not in the table: {code}")
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
                core_mod = {"NAc": "NAc", "NS": "NS"}[n2[0][1]] + core_mod
    return {"stem": stem, "config": config, "ring": ring, "anomer": {"a": "a", "b": "b"}.get(anomer, "?"),
            "anomeric_pos": int(apos), "core_mod": core_mod, "subs": subs}


def _letters(n: int) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [alphabet[i] for i in range(n)]


def parse_wurcs(wurcs: str) -> list[Residue]:
    """Residues of a WURCS 2.0 string as a tree rooted at its first residue (the reducing end)."""
    m = re.match(r"^WURCS=2\.0/(\d+),(\d+),(\d+)/((?:\[[^\]]*\])+)/([^/]*)/(.*)$", wurcs)
    if not m:
        raise ValueError(f"not WURCS 2.0: {wurcs[:60]}")
    n_res = int(m.group(2))
    unique = [decode_wurcs_residue(c) for c in re.findall(r"\[([^\]]*)\]", m.group(4))]
    order = [int(i) - 1 for i in m.group(5).split("-")]
    if len(order) != n_res or n_res > 52:
        raise ValueError("residue count of the WURCS header and its sequence disagree")
    residues = []
    for u in order:
        d = unique[u]
        name, base = make_name(d["stem"], d["config"], d["ring"], d["core_mod"], d["subs"])
        residues.append(Residue(name, base, d["stem"], d["config"], d["ring"], d["anomer"], d["anomeric_pos"]))
    index = {letter: i for i, letter in enumerate(_letters(n_res))}
    adjacency: dict[int, list] = {i: [] for i in range(n_res)}
    for token in (m.group(6).split("_") if m.group(6) else []):
        left, right = token.split("-")
        ends = []
        for side in (left, right):
            alts = side.split("|")
            letters = {re.match(r"^([a-zA-Z])", a).group(1) for a in alts}
            if len(letters) != 1:
                raise ValueError(f"linkage with alternative residues: {token}")
            ends.append((index[letters.pop()], "/".join(a[1:] for a in alts)))
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


# --- GLYCAM sequence names ------------------------------------------------------------------------------------
GLYCAM_RES = re.compile(
    r"(?P<config>[DL])?(?P<stem>Glc|Gal|Man|Fuc|Neu|Xyl|Rha|Ara|Fru|Ido|KDN|KDO|Qui|Rib|Lyx|All|Alt|Gul|Tal|Psi|Tag|Sor|Bac)"
    r"(?P<ring>[pf])(?P<mod>NAc|NS|A|5Ac|5Gc)?(?:\{(?P<subs>[^}]*)\})?(?P<anomer>[ab])(?P<apos>\d)-(?P<to>\d|OH)")
GLYCAM_SUB = {"S": "S", "Me": "Me", "A": "Ac", "Ac": "Ac", "PC": "PCho"}


def parse_glycam(name: str) -> list[Residue]:
    """A GLYCAM condensed sequence (non-reducing end first, ending in `a1-OH` or `b1-OH`) as a tree.

    The returned list is ordered as read; the reducing-end residue is the last one and is the root."""
    text = re.sub(r"\[(\d[A-Za-z]*(?:,\d[A-Za-z]*)*)\]", lambda m: "{" + m.group(1) + "}", name)
    residues: list[Residue] = []
    pending: list[tuple[int, str]] = []
    stack: list[list] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "[":
            stack.append(pending)
            pending = []
            pos += 1
            continue
        if ch == "]":
            pending = stack.pop() + pending
            pos += 1
            continue
        m = GLYCAM_RES.match(text, pos)
        if not m:
            raise ValueError(f"GLYCAM residue not understood at {pos}: {text[pos:pos + 30]}")
        pos = m.end()
        g = m.groupdict()
        stem = {"KDN": "Kdn", "KDO": "Kdo"}.get(g["stem"], g["stem"])
        config = g["config"] or "D"                        # a few names write Neup5Ac without its D
        subs = []
        for s in (g["subs"].split(",") if g["subs"] else []):
            sm = re.match(r"^(\d)([A-Za-z]+)$", s)
            subs.append((int(sm.group(1)), GLYCAM_SUB[sm.group(2)]))
        mod = g["mod"] or ""
        if stem == "Bac":
            mod, subs = "".join(f"{p}{s}" for p, s in sorted(subs)), []
        res = Residue(*make_name(stem, config, g["ring"], mod, subs), stem, config, g["ring"], g["anomer"],
                      int(g["apos"]))
        residues.append(res)
        me = len(residues) - 1
        for child, to in pending:
            residues[child].parent, residues[child].parent_pos = me, to
            res.children.append(child)
        pending = [] if g["to"] == "OH" else [(me, g["to"])]
    if stack or pending:
        raise ValueError("GLYCAM name does not end at a reducing end")
    return residues


# --- what is read from a tree ---------------------------------------------------------------------------------
def root_of(residues: list[Residue]) -> int:
    return next(i for i, r in enumerate(residues) if r.parent is None)


def linkages(residues: list[Residue]) -> list[tuple[str, str, str]]:
    """(donor residue, linkage such as 'b1-4', acceptor residue) for every glycosidic bond."""
    return [(r.name, f"{r.anomer}{r.anomeric_pos}-{r.parent_pos}", residues[r.parent].name)
            for r in residues if r.parent is not None]


def depth(residues: list[Residue]) -> int:
    """Residues on the longest path from the reducing end to a terminal residue."""
    best = 0
    for i, r in enumerate(residues):
        n, j = 1, i
        while residues[j].parent is not None:
            j, n = residues[j].parent, n + 1
        best = max(best, n)
    return best


def condensed(residues: list[Residue], i: int | None = None) -> str:
    """IUPAC-condensed text of the tree, branches sorted (longest chain first), for reading and for matching."""
    i = root_of(residues) if i is None else i

    def size(j):
        return 1 + sum(size(c) for c in residues[j].children)

    def write(j):
        r = residues[j]
        kids = sorted(r.children, key=lambda c: (-size(c), residues[c].parent_pos or "", residues[c].name))
        text = ""
        for n, c in enumerate(kids):
            piece = write(c) + f"({residues[c].anomer}{residues[c].anomeric_pos}-{residues[c].parent_pos})"
            text += piece if n == 0 else f"[{piece}]"
        return text + r.name

    return write(i)


# ==============================================================================================================
# Class rules (coverage_classes.py)
# ==============================================================================================================




SIALIC = {"Neu5Ac", "Neu5Gc", "Kdn"}
URONIC = {"GlcA", "IdoA"}
HEXOSAMINE = {"GlcNAc", "GlcNS", "GalNAc"}
# Residues of vertebrate glycoconjugates; any other residue marks an entry as microbial or plant
VERTEBRATE = {"Glc", "Gal", "Man", "GlcNAc", "GalNAc", "Fuc", "Xyl", "GlcA", "IdoA", "GlcNS"} | SIALIC

CLASSES = [  # key, label on the figure, longer description
    ("n_glycan", "N-glycans", "Man(b1-4)GlcNAc(b1-4)GlcNAc at the reducing end"),
    ("o_galnac", "O-GalNAc glycans", "GalNAc at the reducing end carrying one of the mucin-type cores 1 to 8, or sialyl-Tn"),
    ("gag", "Glycosaminoglycans", "uronic acid and hexosamine in the linkages of heparan sulfate, chondroitin or dermatan sulfate or hyaluronan, or the Xyl linkage region"),
    ("milk", "Milk oligosaccharides", "lactose at the reducing end, bare, fucosylated, sialylated, galactosylated or extended by GlcNAc (lacto and neolacto series)"),
    ("glycolipid", "Glycolipid headgroups", "lactose at the reducing end extended as in the ganglio, globo or isoglobo series"),
    ("o_other", "O-Man, O-Fuc, O-Glc glycans", "Man, Fuc or Glc at the reducing end carrying the first residues of an O-mannose, O-fucose or O-glucose glycan"),
    ("epitope", "Terminal determinants and fragments", "Gal, GlcNAc, GalNAc, Fuc, Glc and sialic acid only, without one of the cores above: Lewis, ABH and LacNAc determinants and smaller fragments"),
    ("microbial", "Microbial and plant glycans", "a residue that vertebrate glycoconjugates do not use, or a homo-oligomer of Glc, Man, Xyl or GlcNAc"),
    ("mono", "Monosaccharides", "a single residue"),
    ("other", "Other", "none of the rules holds"),
]
LABEL = {k: label for k, label, _ in CLASSES}


def sulfate_groups(x: Residue) -> int:
    """O-sulfate groups written after the residue name, plus the N-sulfate of GlcNS."""
    return len(re.findall(r"\dS", x.name[len(x.base):])) + (1 if "NS" in x.base else 0)


class Tree:
    def __init__(self, residues: list[Residue]):
        self.r = residues
        self.root = root_of(residues)
        self.n = len(residues)
        self.bases = [x.base for x in residues]

    def kids(self, i: int, base=None, link: str | None = None) -> list[int]:
        """Children of residue i, optionally with a given base name (or set of names) and linkage ('b1-4')."""
        out = []
        for c in self.r[i].children:
            x = self.r[c]
            if base is not None and x.base not in ({base} if isinstance(base, str) else base):
                continue
            if link is not None and f"{x.anomer}{x.anomeric_pos}-{x.parent_pos}" != link:
                continue
            out.append(c)
        return out

    def link(self, c: int) -> str:
        x = self.r[c]
        return f"{x.anomer}{x.anomeric_pos}-{x.parent_pos}"

    def bonds(self):
        """(donor base, linkage, acceptor base) of every glycosidic bond."""
        return [(x.base, self.link(i), self.r[x.parent].base) for i, x in enumerate(self.r) if x.parent is not None]

    def subtree(self, i: int) -> list[int]:
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(self.r[j].children)
        return out

    def has_sialic(self) -> bool:
        return any(b in SIALIC for b in self.bases)


# --- N-glycans --------------------------------------------------------------------------------------------------
def n_glycan(t: Tree):
    """Root GlcNAc <- GlcNAc(b1-4) <- Man(b1-4). Returns the details of the core, or None."""
    if t.bases[t.root] != "GlcNAc":
        return None
    for g2 in t.kids(t.root, "GlcNAc", "b1-4"):
        for bman in t.kids(g2, "Man", "b1-4"):
            arms = [c for c in t.kids(bman, "Man") if t.r[c].anomer == "a"]       # a1-3 and a1-6 (or left open)
            n_ant = sum(len(t.kids(a, "GlcNAc")) for a in arms)
            n_arm_man = sum(len(t.kids(a, "Man")) for a in arms)
            n_man = sum(1 for i in t.subtree(bman) if t.bases[i] == "Man")
            if n_ant and n_arm_man:
                kind = "hybrid"
            elif n_ant:
                kind = "complex"
            elif n_arm_man:
                kind = "oligomannose"
            else:
                kind = "core"                                   # at most the trimannosyl core: paucimannose
            usual = {("Man", "a1-3"), ("Man", "a1-6"), ("GlcNAc", "b1-4"), ("Xyl", "b1-2")}
            return {
                "kind": kind, "n_man": n_man, "antennae": n_ant,
                "core_fuc6": bool(t.kids(t.root, "Fuc", "a1-6")), "core_fuc3": bool(t.kids(t.root, "Fuc", "a1-3")),
                "bisecting": bool(t.kids(bman, "GlcNAc", "b1-4")), "xylose": bool(t.kids(bman, "Xyl", "b1-2")),
                "sialylated": t.has_sialic(), "sulfated": any(sulfate_groups(x) for x in t.r),
                "foreign_residues": foreign_residues(t),
                "unusual_on_beta_man": sorted(f"{t.r[c].name}({t.link(c)})" for c in t.r[bman].children
                                              if (t.bases[c], t.link(c)) not in usual),
                "open_linkages": sorted({t.link(i) for i, x in enumerate(t.r) if x.parent is not None and "/" in t.link(i)}),
            }
    return None


def n_glycan_like(t: Tree) -> bool:
    """A chitobiose at the reducing end with Man beyond it, but not the canonical Man(b1-4) (used for flagging)."""
    if t.bases[t.root] != "GlcNAc" or n_glycan(t):
        return False
    return any("Man" in {t.bases[i] for i in t.subtree(g2)} for g2 in t.kids(t.root, "GlcNAc", "b1-4"))


N_KIND_LABEL = {"oligomannose": "oligomannose", "hybrid": "hybrid", "complex": "complex",
                "core": "core and paucimannose"}


# --- glycosaminoglycans ------------------------------------------------------------------------------------------
GAG_BONDS = {
    "heparan sulfate / heparin": {("GlcA", "b1-4", "GlcNAc"), ("GlcA", "b1-4", "GlcNS"), ("IdoA", "a1-4", "GlcNAc"),
                                  ("IdoA", "a1-4", "GlcNS"), ("GlcNAc", "a1-4", "GlcA"), ("GlcNAc", "a1-4", "IdoA"),
                                  ("GlcNS", "a1-4", "GlcA"), ("GlcNS", "a1-4", "IdoA")},
    "chondroitin / dermatan sulfate": {("GlcA", "b1-3", "GalNAc"), ("IdoA", "a1-3", "GalNAc"),
                                       ("GalNAc", "b1-4", "GlcA"), ("GalNAc", "b1-4", "IdoA")},
    "hyaluronan": {("GlcA", "b1-3", "GlcNAc"), ("GlcNAc", "b1-4", "GlcA")},
}


def linkage_region(t: Tree) -> list[int] | None:
    """GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl at the reducing end: the residues of the tetrasaccharide, or None."""
    if t.bases[t.root] != "Xyl":
        return None
    for g1 in t.kids(t.root, "Gal", "b1-4"):
        for g2 in t.kids(g1, "Gal", "b1-3"):
            for u in t.kids(g2, "GlcA", "b1-3"):
                return [t.root, g1, g2, u]
    return None


def gag(t: Tree):
    region = linkage_region(t)
    bonds = t.bonds()
    found = {name: sum(1 for b in bonds if b in kinds) for name, kinds in GAG_BONDS.items()}
    found = {k: v for k, v in found.items() if v}
    rest = [i for i in range(t.n) if i not in (region or [])]
    only_gag_residues = all(t.bases[i] in URONIC | HEXOSAMINE for i in rest)
    if region and not found:
        return {"kind": "linkage region", "repeat_bonds": 0, "linkage_region": True,
                "beyond_region": sorted({t.bases[i] for i in rest})}
    if found and (only_gag_residues or region):
        kind = max(found, key=found.get)
        sulfates = sum(sulfate_groups(x) for x in t.r)
        if region and not only_gag_residues:
            kind = "linkage region with " + ("keratan-type chain" if any(t.bases[i] == "Gal" for i in rest) else kind)
        return {"kind": kind, "repeat_bonds": sum(found.values()), "linkage_region": bool(region),
                "sulfated": sulfates > 0, "n_sulfate_groups": sulfates}
    return None


# --- lactose at the reducing end ---------------------------------------------------------------------------------
def lactose(t: Tree):
    """Gal(b1-4)Glc at the reducing end; the series is read from what the lactose Gal carries."""
    if t.bases[t.root] != "Glc":
        return None
    gals = t.kids(t.root, "Gal", "b1-4")
    if not gals:
        return None
    gal = gals[0]
    on_gal = {(t.bases[c], t.link(c)) for c in t.r[gal].children}
    bases_on_gal = {b for b, _ in on_gal}
    fucosylated_gal = ("Fuc", "a1-2") in on_gal
    if ("GalNAc", "b1-4") in on_gal:
        return {"kind": "ganglio series", "family": "glycolipid"}
    sia = [c for c in t.r[gal].children if t.bases[c] in SIALIC]
    if any(t.kids(s, SIALIC) for s in sia):
        return {"kind": "ganglio series (GD3, GT3)", "family": "glycolipid"}
    if ("Gal", "a1-4") in on_gal and not fucosylated_gal:
        return {"kind": "globo series", "family": "glycolipid"}
    if ("Gal", "a1-3") in on_gal and not fucosylated_gal:
        return {"kind": "isoglobo series", "family": "glycolipid"}
    if "GlcNAc" in bases_on_gal:
        first = [c for c in t.r[gal].children if t.bases[c] == "GlcNAc"]
        type1 = any(t.kids(c, "Gal", "b1-3") for c in first)
        type2 = any(t.kids(c, "Gal", "b1-4") for c in first)
        kind = {(True, False): "lacto series (type 1)", (False, True): "neolacto series (type 2)",
                (True, True): "lacto and neolacto branches"}.get((type1, type2), "lacto-N-triose and other GlcNAc extensions")
        return {"kind": kind, "family": "milk", "also_glycolipid_series": True}
    if bases_on_gal <= {"Fuc"} | SIALIC | {"Gal", "GalNAc"} and all(t.bases[c] == "Fuc" or c == gal for c in t.r[t.root].children):
        if not on_gal and len(t.r[t.root].children) == 1:
            kind = "lactose"
        elif bases_on_gal & SIALIC and not bases_on_gal - SIALIC:
            kind = "sialyllactose"
        elif bases_on_gal <= {"Fuc"}:
            kind = "fucosyllactose"
        elif fucosylated_gal:
            kind = "blood-group A, B on lactose"
        elif bases_on_gal <= {"Gal"}:
            kind = "galactosyllactose"
        else:
            kind = "other lactose derivative"
        return {"kind": kind, "family": "milk", "also_glycolipid_series": kind == "sialyllactose"}
    return {"kind": "other lactose derivative", "family": "milk"}


# --- O-linked cores ----------------------------------------------------------------------------------------------
def o_galnac(t: Tree):
    """GalNAc at the reducing end; the core from what it carries at O-3 and O-6 (Brockhausen's cores 1 to 8)."""
    if t.bases[t.root] != "GalNAc" or t.n < 2:
        return None
    on = {}
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
    return {"kind": core, "sialylated": t.has_sialic()}


def o_other(t: Tree):
    root = t.bases[t.root]
    if t.n < 2:
        return None
    if root == "Man" and not t.kids(t.root, "Man"):
        kids = {(t.bases[c], t.link(c)) for c in t.r[t.root].children}
        if kids and kids <= {("GlcNAc", "b1-2"), ("GlcNAc", "b1-6"), ("GlcNAc", "b1-4")}:
            core = "core M3" if ("GlcNAc", "b1-4") in kids else "core M2" if ("GlcNAc", "b1-6") in kids else "core M1"
            return {"kind": f"O-mannose, {core}", "also_n_glycan_antenna": core != "core M3"}
    if root == "Fuc":
        kids = {(t.bases[c], t.link(c)) for c in t.r[t.root].children}
        if kids in ({("GlcNAc", "b1-3")}, {("GlcNAc", "b1-3"), ("GlcA", "b1-4")}):
            return {"kind": "O-fucose (EGF repeats)"}
        if kids == {("Glc", "b1-3")}:
            return {"kind": "O-fucose (thrombospondin repeats)"}
    if root == "Glc":
        for x in t.kids(t.root, "Xyl", "a1-3"):
            if all(t.bases[i] == "Xyl" for i in t.subtree(x)) and len(t.r[t.root].children) == 1:
                return {"kind": "O-glucose (EGF repeats)"}
    if root == "Xyl" and set(t.bases) == {"Xyl", "GlcA"} and t.n > 2:
        if all(b in {("GlcA", "b1-3", "Xyl"), ("Xyl", "a1-3", "GlcA"), ("GlcA", "b1-4", "Xyl")} for b in t.bonds()):
            return {"kind": "matriglycan repeats (O-mannose)"}
    return None


# --- microbial and plant glycans, homo-oligomers --------------------------------------------------------------------
def foreign_residues(t: Tree) -> list[str]:
    """Residues that vertebrate glycoconjugates do not use, including methylated and acetylated neutral sugars."""
    out = set()
    for x in t.r:
        tail = x.name[len(x.base):]
        if x.base not in VERTEBRATE or x.config == "?" or x.ring == "f":
            out.add(x.name)
        elif x.base not in SIALIC and ("Me" in tail or "Ac" in tail or "PCho" in tail):
            out.add(x.name)
    return sorted(out)


HOMO = {"Glc": "glucan", "Man": "mannan", "Xyl": "xylan", "GlcNAc": "N-acetylglucosamine oligomer"}


def microbial(t: Tree):
    foreign = foreign_residues(t)
    if not foreign or t.n < 2:
        return None
    stems = {x.stem + ("f" if x.ring == "f" else "") for x in t.r if x.name in foreign}
    if stems & {"Araf", "Galf"}:
        kind = "arabinofuranose and galactofuranose oligomers"
    elif "Rha" in stems:
        kind = "rhamnose-containing"
    elif "Gal" in stems and any(x.base == "GalA" for x in t.r):
        kind = "galacturonic acid-containing"
    elif "Fruf" in stems:
        kind = "fructan"
    else:
        kind = "other residues"
    return {"kind": kind, "foreign_residues": foreign}


def homo_oligomer(t: Tree):
    kinds = set(t.bases)
    if t.n < 2 or len(kinds) != 1 or t.r[t.root].name != t.r[t.root].base:
        return None
    base = kinds.pop()
    if base not in HOMO:
        return None
    links = sorted({t.link(i) for i, x in enumerate(t.r) if x.parent is not None})
    kind = HOMO[base]
    out = {"kind": f"{kind} ({', '.join(links)})", "links": links}
    if base == "GlcNAc":
        if links == ["b1-4"]:
            out["kind"] = "chitin oligomer (b1-4)"
            if t.n == 2:
                return None                                   # chitobiose: left to the rules below
        elif links == ["b1-6"]:
            out["kind"] = "poly-N-acetylglucosamine (b1-6)"
    if base == "Man" and set(links) <= {"a1-2", "a1-3", "a1-6"}:
        out["also_n_glycan_arm"] = True
    return out


# --- determinants built from vertebrate residues ------------------------------------------------------------------
EPITOPE_RESIDUES = {"Gal", "GlcNAc", "GalNAc", "Fuc", "Glc"} | SIALIC
N_CORE_BONDS = {("GlcNAc", "b1-4", "GlcNAc"), ("Fuc", "a1-6", "GlcNAc"), ("Man", "b1-4", "GlcNAc")}


def epitope(t: Tree):
    if t.n < 2:
        return None
    bases = set(t.bases)
    bonds = t.bonds()
    root = t.bases[t.root]
    if root == "GlcNAc" and t.n <= 3 and set(bonds) <= N_CORE_BONDS:
        return {"kind": "N-glycan core fragment", "also_chitin_dimer": bonds == [("GlcNAc", "b1-4", "GlcNAc")]}
    if ("GlcA", "b1-3", "Gal") in bonds and bases <= EPITOPE_RESIDUES | {"GlcA"} and t.r[t.root].children \
            and sum(1 for b in t.bases if b == "GlcA") == 1 and root != "Gal":
        return {"kind": "glucuronylated LacNAc (HNK-1 type)", "sulfated": any(sulfate_groups(x) for x in t.r)}
    if not bases <= EPITOPE_RESIDUES:
        return None
    if bases <= SIALIC:
        links = sorted({b[1] for b in bonds})
        return {"kind": f"oligosialic acid ({', '.join(links)})"}
    has = lambda *b: any(x in bonds for x in b)  # noqa: E731
    if has(("Fuc", "a1-2", "Gal")) and has(("GalNAc", "a1-3", "Gal"), ("Gal", "a1-3", "Gal")):
        kind = "blood group A, B"
    elif has(("Fuc", "a1-3", "GlcNAc"), ("Fuc", "a1-4", "GlcNAc"), ("Fuc", "a1-3", "Glc")):
        kind = "Lewis"
    elif has(("Fuc", "a1-2", "Gal")):
        kind = "blood group H"
    elif has(("Gal", "b1-4", "GlcNAc"), ("Gal", "b1-3", "GlcNAc"), ("GalNAc", "b1-4", "GlcNAc")):
        kind = "sialylated LacNAc" if bases & SIALIC else "LacNAc and poly-LacNAc"
    elif bases & SIALIC:
        kind = "other sialylated fragment"
    else:
        kind = "other neutral fragment"
    return {"kind": kind, "reducing_end": root, "sialylated": bool(bases & SIALIC),
            "sulfated": any(sulfate_groups(x) for x in t.r)}


# --- the order of the rules ---------------------------------------------------------------------------------------
def _mono(t):
    return {"kind": t.r[t.root].name} if t.n == 1 else None


def _lactose_family(family):
    def rule(t):
        hit = lactose(t)
        return hit if hit and hit["family"] == family else None
    return rule


def _microbial_all(t):
    return microbial(t) or homo_oligomer(t)


RULES = [("mono", _mono), ("n_glycan", n_glycan), ("gag", gag), ("microbial", _microbial_all),
         ("milk", _lactose_family("milk")), ("glycolipid", _lactose_family("glycolipid")), ("o_galnac", o_galnac),
         ("o_other", o_other), ("epitope", epitope)]

RULE_TEXT = {
    "mono": "One residue.",
    "n_glycan": "The reducing-end residue is GlcNAc, it carries GlcNAc(b1-4), and that carries Man(b1-4). Type from the "
                "two alpha-Man arms on the beta-Man: complex when an arm carries GlcNAc and no arm carries a further Man; "
                "hybrid when one arm carries GlcNAc and an arm carries a further Man; oligomannose when the arms carry only "
                "Man (Glc caps allowed); core and paucimannose when neither arm carries anything (at most Man3GlcNAc2, "
                "with or without core Fuc or Xyl). Core fucose: Fuc(a1-6) or Fuc(a1-3) on the reducing GlcNAc. Sialylated: "
                "any Neu5Ac, Neu5Gc or Kdn.",
    "gag": "At least one bond of a glycosaminoglycan repeat (heparan sulfate or heparin: GlcA(b1-4)GlcN, IdoA(a1-4)GlcN, "
           "GlcN(a1-4)GlcA or IdoA; chondroitin or dermatan sulfate: GlcA(b1-3)GalNAc, IdoA(a1-3)GalNAc, GalNAc(b1-4)GlcA "
           "or IdoA; hyaluronan: GlcA(b1-3)GlcNAc, GlcNAc(b1-4)GlcA; GlcN is GlcNAc or GlcNS, sulfate groups ignored) and "
           "no residue other than uronic acids and hexosamines outside the linkage region, or the linkage region "
           "GlcA(b1-3)Gal(b1-3)Gal(b1-4)Xyl at the reducing end.",
    "microbial": "More than one residue, and either a residue that vertebrate glycoconjugates do not use (anything other "
                 "than Glc, Gal, Man, GlcNAc, GalNAc, Fuc, Xyl, GlcA, IdoA, GlcNS, Neu5Ac, Neu5Gc, Kdn in their common "
                 "pyranose forms; furanoses; methylated or acetylated neutral sugars), or a homo-oligomer of Glc, Man, Xyl "
                 "or GlcNAc (chitobiose excepted).",
    "milk": "Gal(b1-4)Glc at the reducing end, and the lactose Gal is bare, fucosylated, sialylated once, galactosylated, "
            "or extended by GlcNAc (lacto series: Gal(b1-3)GlcNAc; neolacto series: Gal(b1-4)GlcNAc).",
    "glycolipid": "Gal(b1-4)Glc at the reducing end, and the lactose Gal carries GalNAc(b1-4) (ganglio series), an "
                  "oligosialyl chain (GD3, GT3), Gal(a1-4) (globo series) or Gal(a1-3) (isoglobo series) without Fuc(a1-2).",
    "o_galnac": "GalNAc at the reducing end, substituted only at O-3 and O-6: core 1 Gal(b1-3); core 2 Gal(b1-3) and "
                "GlcNAc(b1-6); core 3 GlcNAc(b1-3); core 4 GlcNAc(b1-3) and GlcNAc(b1-6); core 5 GalNAc(a1-3); core 6 "
                "GlcNAc(b1-6); core 7 GalNAc(a1-6); core 8 Gal(a1-3); sialyl-Tn sialic acid (a2-6) alone. A sialic acid at "
                "O-6 is allowed on cores 1, 3 and 5.",
    "o_other": "Man at the reducing end carrying only GlcNAc(b1-2), GlcNAc(b1-6) or GlcNAc(b1-4) (O-mannose cores M1, M2, "
               "M3); Fuc carrying GlcNAc(b1-3) or Glc(b1-3) (O-fucose); Glc carrying a chain of Xyl(a1-3) (O-glucose); or "
               "the Xyl-GlcA repeats of matriglycan.",
    "epitope": "More than one residue, only Gal, GlcNAc, GalNAc, Fuc, Glc and sialic acids (sulfate allowed), and none of "
               "the rules above. Named by the first that holds: N-glycan core fragment (at most three residues of "
               "GlcNAc(b1-4)GlcNAc, Fuc(a1-6)GlcNAc, Man(b1-4)GlcNAc); glucuronylated LacNAc (one GlcA(b1-3)Gal, HNK-1 "
               "type); oligosialic acid (sialic acids only); blood group A or B (Fuc(a1-2)Gal with GalNAc(a1-3) or "
               "Gal(a1-3)); Lewis (Fuc(a1-3) or Fuc(a1-4) on GlcNAc or Glc); blood group H (Fuc(a1-2)Gal); LacNAc "
               "(Gal(b1-3), Gal(b1-4) or GalNAc(b1-4) on GlcNAc); other sialylated or neutral fragments.",
    "other": "None of the rules holds.",
}


def classify(residues: list[Residue]) -> dict:
    """Class, subclass and details of one entry, and the other classes whose rule also holds."""
    t = Tree(residues)
    hits = [(key, rule(t)) for key, rule in RULES]
    hits = [(k, v) for k, v in hits if v]
    if not hits:
        like = n_glycan_like(t)
        return {"class": "other", "subclass": "N-glycan with a core that is not Man(b1-4)GlcNAc(b1-4)GlcNAc" if like else "",
                "detail": {"n_glycan_like": like}, "also": []}
    key, detail = hits[0]
    subclass = N_KIND_LABEL[detail["kind"]] if key == "n_glycan" else detail["kind"]
    # the determinant rule is the remainder of the vertebrate residues, so it is not counted as a second class
    return {"class": key, "subclass": subclass, "detail": detail, "also": [k for k, _ in hits[1:] if k != "epitope"]}
