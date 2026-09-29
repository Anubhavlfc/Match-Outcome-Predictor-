"""Canonical team names.

Different sources spell clubs differently ("Man United", "Manchester Utd",
"Manchester United FC"). Every name is mapped to one canonical spelling
before any feature is computed, otherwise one club would be split into
several "teams" with separate histories.

Unknown names raise an error on purpose: silently passing an unmapped
spelling through would create a phantom team with no history.
"""

from __future__ import annotations

from collections.abc import Iterable

# canonical name -> known aliases (matching is case-insensitive)
_CANONICAL_ALIASES: dict[str, list[str]] = {
    "Arsenal": ["Arsenal FC"],
    "Aston Villa": ["Villa", "Aston Villa FC"],
    "Bournemouth": ["AFC Bournemouth"],
    "Brentford": ["Brentford FC"],
    "Brighton": ["Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton Hove Albion"],
    "Burnley": ["Burnley FC"],
    "Cardiff City": ["Cardiff"],
    "Chelsea": ["Chelsea FC"],
    "Coventry City": ["Coventry"],
    "Crystal Palace": ["Palace"],
    "Everton": ["Everton FC"],
    "Fulham": ["Fulham FC"],
    "Huddersfield Town": ["Huddersfield"],
    "Hull City": ["Hull"],
    "Ipswich Town": ["Ipswich"],
    "Leeds United": ["Leeds"],
    "Leicester City": ["Leicester"],
    "Liverpool": ["Liverpool FC"],
    "Luton Town": ["Luton"],
    "Manchester City": ["Man City", "Manchester City FC", "Man. City"],
    "Manchester United": ["Man United", "Man Utd", "Manchester Utd", "Manchester United FC", "Man. United"],
    "Middlesbrough": ["Boro"],
    "Newcastle United": ["Newcastle", "Newcastle Utd"],
    "Norwich City": ["Norwich"],
    "Nottingham Forest": ["Nott'm Forest", "Nottm Forest", "Nottingham", "Forest"],
    "Queens Park Rangers": ["QPR"],
    "Sheffield United": ["Sheffield Utd", "Sheff Utd"],
    "Southampton": ["Southampton FC"],
    "Stoke City": ["Stoke"],
    "Sunderland": ["Sunderland AFC"],
    "Swansea City": ["Swansea"],
    "Tottenham Hotspur": ["Tottenham", "Spurs"],
    "Watford": ["Watford FC"],
    "West Bromwich Albion": ["West Brom", "West Bromwich"],
    "West Ham United": ["West Ham"],
    "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
}

_LOOKUP: dict[str, str] = {}
for _canonical, _aliases in _CANONICAL_ALIASES.items():
    for _name in [_canonical, *_aliases]:
        _LOOKUP[_name.strip().casefold()] = _canonical


class UnknownTeamError(ValueError):
    """Raised when a team name has no canonical mapping."""


# Club-type affixes some sources add ("Arsenal FC", "AFC Bournemouth", "Hull City AFC").
_SUFFIXES = (" fc", " afc")
_PREFIXES = ("afc ",)


def normalize_team_name(name: str) -> str:
    """Return the canonical spelling of a team name."""
    key = " ".join(str(name).split()).casefold()
    if key in _LOOKUP:
        return _LOOKUP[key]
    stripped = key
    for suffix in _SUFFIXES:
        stripped = stripped.removesuffix(suffix)
    for prefix in _PREFIXES:
        stripped = stripped.removeprefix(prefix)
    if stripped in _LOOKUP:
        return _LOOKUP[stripped]
    raise UnknownTeamError(
        f"Unknown team name {name!r}. Add it to _CANONICAL_ALIASES in src/utils/team_names.py."
    )


def normalize_team_names(names: Iterable[str]) -> list[str]:
    return [normalize_team_name(name) for name in names]


def canonical_teams() -> list[str]:
    return sorted(_CANONICAL_ALIASES)
