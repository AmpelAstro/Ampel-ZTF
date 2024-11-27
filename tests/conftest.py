pytest_plugins = ["tests.fixtures"]


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run docker-based integration tests",
    )
