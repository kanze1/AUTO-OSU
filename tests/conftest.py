"""Keep generation tests out of the user's local record store."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_records(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AUTOOSU_RECORDS", str(tmp_path_factory.mktemp("generation-records")))
        yield
