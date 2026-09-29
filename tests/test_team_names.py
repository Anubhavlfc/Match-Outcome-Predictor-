import pytest

from src.utils.team_names import UnknownTeamError, normalize_team_name


@pytest.mark.parametrize("alias", ["Man United", "Man Utd", "Manchester Utd", "Manchester United", " man united "])
def test_manchester_united_aliases_collapse(alias):
    assert normalize_team_name(alias) == "Manchester United"


@pytest.mark.parametrize("alias, canonical", [
    ("Man City", "Manchester City"),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Spurs", "Tottenham Hotspur"),
    ("Wolves", "Wolverhampton Wanderers"),
    ("West Brom", "West Bromwich Albion"),
    ("Brighton & Hove Albion", "Brighton"),
])
def test_other_aliases(alias, canonical):
    assert normalize_team_name(alias) == canonical


def test_city_and_united_stay_distinct():
    assert normalize_team_name("Man City") != normalize_team_name("Man United")


def test_unknown_team_raises():
    with pytest.raises(UnknownTeamError):
        normalize_team_name("Real Madrid")


def test_every_real_team_name_is_mapped(real_matches):
    # build_match_table would already have raised on an unknown name; also
    # check that each season has exactly 20 distinct canonical teams.
    for _, season in real_matches.groupby("season"):
        assert season[["home_team", "away_team"]].stack().nunique() == 20
