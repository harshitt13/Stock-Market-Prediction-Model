"""Mutation tests for the leakage check: does it actually fail on leakage?

A test that never fails is not a test. ``test_leakage.py`` asserts the current
feature pipeline is clean, which is worth nothing unless the check has the
power to detect a leak that is really there. This module injects known leaks
into a copy of ``fetch_data`` and asserts each one is caught.

The origin of this suite is worth recording. Running the truncation check
against commit 3933168's ``engineer_features`` -- the "before" state, believed
to be leaky -- it *passed*. That was correct: master's backward fill lived in
``fetch_stock_data``, not in ``engineer_features``, so the check never saw it.
But it meant the check's detection power had never actually been demonstrated
on anything. This module demonstrates it.

Mutants are loaded as standalone modules from a rewritten source file, which
works only because ``fetch_data`` has no heavy top-level imports.
"""

import importlib.util
from pathlib import Path

import pytest

import fetch_data
from fetch_data import FEATURE_COLUMNS
from test_leakage import (
    LEADING_TRUNCATION_POINTS,
    TRUNCATION_POINTS,
    fixture_with_leading_macro_gap,
    load_fixture,
    truncation_violations,
)

GAPPED_POINTS = LEADING_TRUNCATION_POINTS + TRUNCATION_POINTS


class Mutation:
    """One injected leak, and where it should be visible."""

    def __init__(self, name, edits, gapped=False, points=None):
        self.name = name
        self.edits = edits
        self.gapped = gapped
        self.points = points or (GAPPED_POINTS if gapped else TRUNCATION_POINTS)

    def fixture(self):
        return fixture_with_leading_macro_gap() if self.gapped else load_fixture()

    def __repr__(self):
        return self.name


MUTATIONS = [
    Mutation(
        "macro_backward_fill",
        [("            df[name] = df[name].ffill()",
          "            df[name] = df[name].ffill().bfill()")],
        gapped=True,
    ),
    Mutation(
        "centred_rolling_sma20",
        [("    sma_20 = close.rolling(window=20).mean()",
          "    sma_20 = close.rolling(window=20, center=True).mean()")],
    ),
    Mutation(
        "centred_rolling_volume",
        [("        avg_volume = volume.rolling(window=20).mean()",
          "        avg_volume = volume.rolling(window=20, center=True).mean()")],
    ),
    Mutation(
        "global_fit_volatility",
        [("    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()",
          "    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()"
          " / df['Log_Return'].std()")],
    ),
    Mutation(
        "direct_future_peek",
        [("    df['hl_range'] = (high - low) / close",
          "    df['hl_range'] = (high - low) / close.shift(-1)")],
    ),
    Mutation(
        "expanding_zscore_rsi",
        [("    df['RSI_14'] = compute_rsi(close, window=14)",
          "    df['RSI_14'] = compute_rsi(close, window=14)"
          " - compute_rsi(close, window=14).mean()")],
    ),
]


def source_text() -> str:
    return Path(fetch_data.__file__).read_text(encoding="utf-8")


def load_mutant(tmp_path: Path, mutation: Mutation):
    """Import a copy of fetch_data with the mutation's edits applied."""
    source = source_text()
    for old, new in mutation.edits:
        assert old in source, (
            f"mutation {mutation.name!r} anchor no longer present in "
            f"fetch_data.py: {old!r}. Update the anchor, do not delete the "
            "mutation."
        )
        source = source.replace(old, new, 1)

    path = tmp_path / f"mutant_{mutation.name}.py"
    path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(f"mutant_{mutation.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_every_anchor_still_exists(mutation):
    """A refactor that renames a mutated line must fail loudly here.

    Otherwise the mutation silently stops being applied and this whole suite
    starts passing vacuously.
    """
    source = source_text()
    for old, _ in mutation.edits:
        assert old in source, f"{mutation.name}: stale anchor {old!r}"


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_injected_leak_is_caught(tmp_path, mutation):
    """Each known leak must produce at least one truncation violation."""
    mutant = load_mutant(tmp_path, mutation)
    violations = truncation_violations(
        mutant.engineer_features, mutation.fixture(), mutation.points, FEATURE_COLUMNS
    )
    assert violations, (
        f"{mutation.name} went UNDETECTED. The leakage check does not "
        "constrain this class of bug."
    )


def test_the_unmutated_module_is_clean(tmp_path):
    """Control. If this fails the mutation harness itself is broken."""
    control = load_mutant(tmp_path, Mutation("control", []))
    for gapped in (False, True):
        raw = fixture_with_leading_macro_gap() if gapped else load_fixture()
        points = GAPPED_POINTS if gapped else TRUNCATION_POINTS
        violations = truncation_violations(
            control.engineer_features, raw, points, FEATURE_COLUMNS
        )
        assert not violations, (gapped, violations[:5])


def test_mutants_are_actually_different_from_the_original(tmp_path):
    """Guards against an edit silently becoming a no-op."""
    original = source_text()
    for mutation in MUTATIONS:
        path = tmp_path / f"check_{mutation.name}.py"
        mutated = original
        for old, new in mutation.edits:
            mutated = mutated.replace(old, new, 1)
        path.write_text(mutated, encoding="utf-8")
        assert mutated != original, f"{mutation.name} is a no-op edit"


def test_backward_fill_is_invisible_without_a_leading_gap(tmp_path):
    """Documents the blind spot that made the source-scan lint necessary.

    ``ffill()`` runs before ``bfill()``, so a backward fill can only ever alter
    rows before the first macro observation. On the ungapped fixture it is a
    no-op and the truncation check cannot see it -- which is precisely why
    ``test_leakage.py`` uses a gapped fixture and low truncation points for
    that case, rather than scanning the source for the string "bfill".
    """
    mutation = MUTATIONS[0]
    assert mutation.name == "macro_backward_fill"
    mutant = load_mutant(tmp_path, mutation)

    invisible = truncation_violations(
        mutant.engineer_features, load_fixture(), TRUNCATION_POINTS, FEATURE_COLUMNS
    )
    assert not invisible, (
        "the ungapped fixture now detects bfill; if that is intentional, this "
        "test documents a stale assumption and should be updated"
    )

    visible = truncation_violations(
        mutant.engineer_features,
        fixture_with_leading_macro_gap(),
        GAPPED_POINTS,
        FEATURE_COLUMNS,
    )
    assert visible, "the gapped fixture must expose what the plain one hides"
