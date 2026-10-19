# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""License normalization + installed-dependency inventory (compliance)."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "DEFAULT_ALLOWLIST",
    "LicenseEntry",
    "dependency_inventory",
    "normalize_license",
]

DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Python-2.0",
        "PSF-2.0",
        "MPL-2.0",
    }
)

# Explicit, case-insensitive alias table.  Keys are anchored regexes matched
# with ``fullmatch`` (never substring search), so e.g. "permitted" cannot be
# mistaken for MIT.  More specific patterns win.
_ALIASES: dict[str, str] = {
    # MIT family.
    r"mit[\s-]+no[\s-]+attribution(?:[\s-]+licen[cs]e)?": "MIT-0",
    r"mit[\s-]*0(?:\.0)?": "MIT-0",
    r"(?:the[\s-]+)?mit(?:[\s-]+licen[cs]e)?": "MIT",
    # BSD family.
    r"(?:bsd[\s-]*2[\s-]*clause|simplified[\s-]+bsd|freebsd)" r"(?:[\s-]+licen[cs]e)?": "BSD-2-Clause",
    r"(?:bsd[\s-]*3[\s-]*clause|3[\s-]*clause[\s-]+bsd|(?:new|modified|revised)[\s-]+bsd|bsd)" r"(?:[\s-]+licen[cs]e)?": "BSD-3-Clause",
    # Apache.
    r"apache(?:[\s-]+software)?[\s-]+licen[cs]e" r"(?:,?[\s-]+(?:version[\s-]+)?2(?:\.0)?)?": "Apache-2.0",
    r"(?:apache|asl)[\s-]*2(?:\.0)?": "Apache-2.0",
    # ISC.
    r"isc(?:[\s-]+licen[cs]e)?": "ISC",
    # Mozilla Public License.
    r"(?:mozilla[\s-]+public[\s-]+licen[cs]e|mpl)" r"(?:,?[\s-]+(?:(?:version|v)\.?[\s-]+)?2(?:\.0)?)?": "MPL-2.0",
    # GNU GPL 2/3, only and or-later ("+" is the deprecated or-later form).
    r"(?:gnu[\s-]+general[\s-]+public[\s-]+licen[cs]e|gnu[\s-]+gpl|gpl)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?2(?:\.0)?(?:[\s-]+only)?": "GPL-2.0-only",
    r"(?:gnu[\s-]+general[\s-]+public[\s-]+licen[cs]e|gnu[\s-]+gpl|gpl)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?2(?:\.0)?[\s-]*(?:\+|or[\s-]+(?:any[\s-]+)?later)": "GPL-2.0-or-later",
    r"(?:gnu[\s-]+general[\s-]+public[\s-]+licen[cs]e|gnu[\s-]+gpl|gpl)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?3(?:\.0)?(?:[\s-]+only)?": "GPL-3.0-only",
    r"(?:gnu[\s-]+general[\s-]+public[\s-]+licen[cs]e|gnu[\s-]+gpl|gpl)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?3(?:\.0)?[\s-]*(?:\+|or[\s-]+(?:any[\s-]+)?later)": "GPL-3.0-or-later",
    # GNU LGPL / AGPL.
    r"(?:lgpl|gnu[\s-]+lesser[\s-]+general[\s-]+public[\s-]+licen[cs]e)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?2\.1(?:[\s-]+only)?": "LGPL-2.1-only",
    r"(?:lgpl|gnu[\s-]+lesser[\s-]+general[\s-]+public[\s-]+licen[cs]e)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?3(?:\.0)?(?:[\s-]+only)?": "LGPL-3.0-only",
    r"(?:agpl|gnu[\s-]+affero[\s-]+general[\s-]+public[\s-]+licen[cs]e)" r"[\s-]*,?[\s-]*(?:v|version[\s-]+)?3(?:\.0)?(?:[\s-]+only)?": "AGPL-3.0-only",
    # Public domain / permissive.
    r"(?:the[\s-]+)?unlicen[cs]e": "Unlicense",
    r"(?:cc[\s-]*0(?:[\s-]+1(?:\.0)?)?|creative[\s-]+commons[\s-]+zero(?:[\s-]+1(?:\.0)?)?)" r"(?:[\s-]+universal)?": "CC0-1.0",
    # Creative Commons.
    r"(?:cc[\s-]*by[\s-]*4(?:\.0)?|creative[\s-]+commons[\s-]+attribution[\s-]+4(?:\.0)?" r"(?:[\s-]+international)?)": "CC-BY-4.0",
    r"(?:cc[\s-]*by[\s-]*nc[\s-]*4(?:\.0)?|creative[\s-]+commons[\s-]+attribution" r"[\s-]+non[\s-]*commercial[\s-]+4(?:\.0)?(?:[\s-]+international)?)": "CC-BY-NC-4.0",
    r"(?:cc[\s-]*by[\s-]*sa[\s-]*4(?:\.0)?|creative[\s-]+commons[\s-]+attribution" r"[\s-]+share[\s-]*alike[\s-]+4(?:\.0)?(?:[\s-]+international)?)": "CC-BY-SA-4.0",
    # Python Software Foundation.
    r"psf(?:[\s-]+licen[cs]e)?(?:[\s-]*,?[\s-]*(?:v|version[\s-]+)?2(?:\.0)?)?": "PSF-2.0",
    r"python[\s-]+software[\s-]+foundation[\s-]+licen[cs]e" r"(?:,?[\s-]+(?:v|version[\s-]+)?2(?:\.0)?)?": "PSF-2.0",
    r"python(?:[\s-]+licen[cs]e)?[\s-]*,?[\s-]*(?:v|version[\s-]+)?2(?:\.0)?": "Python-2.0",
    # Non-SPDX license still tracked explicitly by the registry.
    r"dino[\s-]*v?3(?:[\s-]+licen[cs]e)?": "DINOv3 License",
}

_ATOM_PUNCTUATION = " \t\r\n.,;:!?\"'()[]{}"
_SPDX_OPERATOR = re.compile(r"(?<![\w-])(AND|OR|WITH)(?![\w-])", re.IGNORECASE)
_COMPILED_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    sorted(
        ((re.compile(pattern, re.IGNORECASE), spdx) for pattern, spdx in _ALIASES.items()),
        key=lambda item: len(item[0].pattern),
        reverse=True,
    )
)


@dataclass(frozen=True)
class LicenseEntry:
    name: str
    version: str
    license: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "license": self.license}


def _match_atom(atom: str) -> str | None:
    """Return the canonical id for one license atom, or None if unknown."""
    cleaned = atom.strip(_ATOM_PUNCTUATION)
    if not cleaned:
        return None
    for pattern, spdx in _COMPILED_ALIASES:
        if pattern.fullmatch(cleaned):
            return spdx
    return None


def normalize_license(value: str | None) -> str:
    """Map a license name or SPDX expression to canonical identifiers.

    ``AND``/``OR``/``WITH`` are preserved in order.  Every atom must resolve
    through the explicit alias table; if any atom is unknown the whole value
    becomes ``"UNKNOWN"`` so compliance never silently allows it.
    """
    if not value:
        return "UNKNOWN"
    text = " ".join(str(value).split())
    if not text:
        return "UNKNOWN"

    # Exact whole-string match first: real license names may contain operator
    # words or hyphens (e.g. "GPL-2.0-or-later", "GNU ... v3.0 or later").
    atom = _match_atom(text)
    if atom is not None:
        return atom

    tokens: list[str] = []
    for index, part in enumerate(_SPDX_OPERATOR.split(text)):
        if index % 2:
            tokens.append(part.upper())
            continue
        atom = _match_atom(part)
        if atom is None:
            return "UNKNOWN"
        tokens.append(atom)
    return " ".join(tokens)


def _license_from_metadata(metadata: Any) -> str:
    expression = metadata.get("License-Expression")
    if expression:
        return normalize_license(expression)
    declared = metadata.get("License")
    if declared:
        return normalize_license(declared)
    classifiers = metadata.get_all("Classifier") or []
    for classifier in classifiers:
        if classifier.startswith("License ::"):
            return normalize_license(classifier.split("::")[-1])
    return "UNKNOWN"


def dependency_inventory(
    *, skip: Iterable[str] = ()
) -> list[LicenseEntry]:
    """Installed distributions with their declared license (sorted)."""
    skipped = {name.lower().replace("_", "-") for name in skip}
    entries: dict[tuple[str, str], LicenseEntry] = {}
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = metadata.get("Name")
        if not name:
            continue
        normalized_name = name.lower().replace("_", "-")
        if normalized_name in skipped:
            continue
        version = metadata.get("Version") or ""
        entries[(name, version)] = LicenseEntry(
            name=name,
            version=version,
            license=_license_from_metadata(metadata),
        )
    return [entries[key] for key in sorted(entries)]
