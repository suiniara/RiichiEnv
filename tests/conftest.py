import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--allow-parameter-updates",
        action="store_true",
        default=False,
        help="Explicitly opt into legacy diagnostic tests containing backward/parameter updates",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--allow-parameter-updates"):
        return
    forbidden = pytest.mark.skip(
        reason="Backward/update tests require --allow-parameter-updates; excluded from functional acceptance"
    )
    for item in items:
        if item.get_closest_marker("parameter_updates") is not None:
            item.add_marker(forbidden)
