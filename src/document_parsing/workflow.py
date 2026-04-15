import os
from pathlib import Path
from typing import Annotated, Literal

from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource


class ParseDocumentCostEffectiveEvent(Event):
    pass


class ParseDocumentAgenticEvent(Event):
    pass


class ParseDocumentAgenticPlusEvent(Event):
    pass


# required for all llama cloud calls
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# get this in case running against a different environment than production
LLAMA_CLOUD_BASE_URL = os.getenv("LLAMA_CLOUD_BASE_URL")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_DEPLOY_PROJECT_ID")


async def get_llama_cloud_client(*args, **kwargs) -> AsyncLlamaCloud:
    return AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        base_url=LLAMA_CLOUD_BASE_URL,
        default_headers={"Project-Id": LLAMA_CLOUD_PROJECT_ID}
        if LLAMA_CLOUD_PROJECT_ID
        else {},
    )


class DocumentProcessingState(BaseModel):
    document_path: str = ""


async def _parse_document(
    client: AsyncLlamaCloud,
    document_path: str,
    tier: Literal["fast", "cost_effective", "agentic", "agentic_plus"],
) -> str:
    with Path(document_path).open("rb") as f:
        result = await client.parsing.parse(
            tier=tier,
            version="latest",
            upload_file=(Path(document_path).name, f.read()),
            expand=["markdown_full"],
        )
    return result.markdown_full or ""


class DocumentProcessingWorkflow(Workflow):
    @step
    async def choose_document_parsing_mode(
        self, ev: StartEvent, ctx: Context[DocumentProcessingState]
    ) -> (
        ParseDocumentCostEffectiveEvent
        | ParseDocumentAgenticEvent
        | ParseDocumentAgenticPlusEvent
    ):
        async with ctx.store.edit_state() as state:
            state.document_path = ev.document_path
        if ev.parsing_mode == "cost_effective":
            return ParseDocumentCostEffectiveEvent()
        elif ev.parsing_mode == "agentic":
            return ParseDocumentAgenticEvent()
        else:
            return ParseDocumentAgenticPlusEvent()

    @step
    async def parse_document_cost_effective(
        self,
        ev: ParseDocumentCostEffectiveEvent,
        ctx: Context[DocumentProcessingState],
        client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
    ) -> StopEvent:
        state = await ctx.store.get_state()
        text = await _parse_document(client, state.document_path, "cost_effective")
        return StopEvent(result=text)

    @step
    async def parse_document_agentic(
        self,
        ev: ParseDocumentAgenticEvent,
        ctx: Context[DocumentProcessingState],
        client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
    ) -> StopEvent:
        state = await ctx.store.get_state()
        text = await _parse_document(client, state.document_path, "agentic")
        return StopEvent(result=text)

    @step
    async def parse_document_agentic_plus(
        self,
        ev: ParseDocumentAgenticPlusEvent,
        ctx: Context[DocumentProcessingState],
        client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
    ) -> StopEvent:
        state = await ctx.store.get_state()
        text = await _parse_document(client, state.document_path, "agentic_plus")
        return StopEvent(result=text)


async def main(document_path: str, parsing_mode: str) -> None:
    wf = DocumentProcessingWorkflow(
        timeout=1800
    )  # allow processing jobs up to 30 minutes
    result = await wf.run(document_path=document_path, parsing_mode=parsing_mode)
    print(str(result))


workflow = DocumentProcessingWorkflow(timeout=None)

if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Document path", required=True)
    parser.add_argument(
        "-m",
        "--mode",
        help="Parsing Mode",
        choices=["cost_effective", "agentic", "agentic_plus"],
        required=False,
        default="agentic",
    )
    args = parser.parse_args()

    if not os.getenv("LLAMA_CLOUD_API_KEY", None):
        raise ValueError(
            "You need to set LLAMA_CLOUD_API_KEY in your environment before using this workflow"
        )

    asyncio.run(main(args.path, args.mode))
