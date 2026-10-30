import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


@pytest.fixture(scope="module")
def run():
    for script in ("generate.py", "score.py"):
        subprocess.run([sys.executable, str(ROOT / "pipeline" / script)], check=True,
                       capture_output=True)
    book = pd.read_csv(ROOT / "results" / "synthetic_member_book.csv")
    res = json.loads((ROOT / "results" / "results.json").read_text())
    return book, res


def test_book_is_labelled_synthetic(run):
    book, _ = run
    assert (book.data_label == "SYNTHETIC").all()


def test_book_matches_published_summary(run):
    book, res = run
    assert len(book) == 490
    assert int(book.scorable.sum()) == 374
    assert int((book.loan_amount_etb > 0).sum()) == 50
    assert int(book.default_flag.sum()) == 0
    assert int(book.penalty_postings.sum()) == 115
    assert int((book.penalty_postings > 0).sum()) == 31


def test_generator_is_deterministic(run):
    book, _ = run
    first = book.copy()
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "generate.py")], check=True,
                   capture_output=True)
    again = pd.read_csv(ROOT / "results" / "synthetic_member_book.csv")
    pd.testing.assert_frame_equal(first, again)


def test_metrics_are_in_range(run):
    _, res = run
    for k in ("gini_all", "gini_borrowers", "gini_all_cv", "gini_borrowers_cv"):
        assert -1 <= res[k] <= 1
    assert 0 <= res["ks_all"] <= 1
    assert res["zero_default_bound"] == pytest.approx(1 - 0.1 ** (1 / 50))


def test_gini_and_ks_on_known_cases():
    from score import gini, ks
    y = np.array([0, 0, 0, 1, 1])
    assert gini(y, np.array([1, 2, 3, 4, 5])) == pytest.approx(1.0)
    assert gini(y, -np.array([1, 2, 3, 4, 5])) == pytest.approx(-1.0)
    assert ks(y, np.array([1, 2, 3, 4, 5])) == pytest.approx(1.0)
