"""Unit tests for response supervision pass-disproof provenance rules."""

import pytest

from riichienv.kar.response_supervision import unique_pass_provenance


@pytest.mark.parametrize(
    "ids,result,proven",
    [
        ((221, 222), "pon", False),
        ((221, 312), "pon", False),
        ((221, 312), "chi", True),
        ((221, 312, 222), "chi", False),
        ((221, 543), "pon", True),
        ((221, 543), "replacement_draw", True),
        ((221, 543), "all_ron_results", True),
        ((221, 543, 312), "all_ron_results", False),
        ((221, 543), "no_ron_draw/exhaustive_draw", True),
        ((221, 543), "ron_results", False),
        ((221, 543), "unresolved", False),
        ((221,), "next_draw", False),
    ],
)
def test_only_disproved_nonpass_alternatives_are_eliminated(ids, result, proven):
    assert bool(unique_pass_provenance(ids, result)) is proven
