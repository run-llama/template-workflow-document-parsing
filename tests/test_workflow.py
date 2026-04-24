import pytest
from document_parsing.workflow import workflow
from llama_cloud_fake import FakeLlamaCloudServer


@pytest.mark.parametrize("parsing_mode", ["cost_effective", "agentic", "agentic_plus"])
async def test_document_parsing_workflow(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
    parsing_mode: str,
) -> None:
    """Exercise parsing.parse against the fake for each tier."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    result = await workflow.run(
        document_path="tests/files/test.pdf",
        parsing_mode=parsing_mode,
    )
    # The SDK call succeeded and returned markdown_full as a string.
    # The fake's payload may be empty; we only care the v2 `parsing.parse`
    # surface is wired correctly.
    assert isinstance(result, str)
