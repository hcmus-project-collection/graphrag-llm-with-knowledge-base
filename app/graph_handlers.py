import json
import logging
from functools import lru_cache

import httpx
import json_repair
from pydantic import BaseModel, model_validator

from . import constants as const
from .models import ResponseMessage
from .utils import limit_asyncio_concurrency

logger = logging.getLogger(__name__)


@limit_asyncio_concurrency(const.MAX_NUM_CONCURRENT_LLM_CALL * 1.5)
async def call_llm_prioritized(messages: list[dict[str, str]]) -> str:
    """Call large language model with prioritized messages."""
    payload = {
        "model": const.MODEL_NAME,
        "messages": messages,
        "temperature": const.DEFAULT_LLM_TEMPERATURE,
        "seed": const.DEFAULT_LLM_SEED,
        "max_tokens": const.DEFAULT_LLM_MAX_TOKENS,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {const.LLM_API_KEY}",
    }

    logger.debug(f"Payload: {payload}")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            const.LLM_API_BASE + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(300),
        )

    if response.status_code != 200:
        logger.debug(f"Response: {response.text}")
        return None

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]

    return content


@limit_asyncio_concurrency(const.MAX_NUM_CONCURRENT_LLM_CALL)
async def call_llm(messages: list[dict[str, str]]) -> str:
    """Call large language model."""
    return await call_llm_prioritized(messages)


class Triplet(BaseModel):
    """Implementation of Triplet class."""

    s1: str
    s2: str
    relation: str

    def fact(self) -> str:
        """Return the fact."""
        return f"{self.s1} {self.relation} {self.s2}"

    @model_validator(mode="before")
    def from_list(self, data: list[str] | dict[str, str]) -> dict[str, str]:
        """Construct triplet from a list or dictionary."""
        if isinstance(data, list):
            assert len(data) >= 3 and all(  # noqa: PT018, S101
                isinstance(s, str) for s in data[:3]
            ), "The list of data must present at least 3 string values"

            return {
                "s1": data[0],
                "relation": data[1],
                "s2": data[2],
            }

        assert all(  # noqa: S101
            k in data and isinstance(data[k], str)
            for k in ["s1", "s2", "relation"]
        ), "Missing key(s) to construct triplet. Requires s1, s2 and relation"

        return {
            "s1": data["s1"],
            "relation": data["relation"],
            "s2": data["s2"],
        }


class GraphKnowledge:
    """Implementation of GraphKnowledge class."""

    def __init__(
        self,
        graph_system_prompt: str = const.GRAPH_SYSTEM_PROMPT,
        ner_system_prompt: str = const.NER_SYSTEM_PROMPT,
        refine_query_system_prompt: str = const.REFINE_QUERY_SYSTEM_PROMPT,
    ) -> None:
        self.graph_system_prompt: str = graph_system_prompt
        self.ner_system_prompt: str = ner_system_prompt
        self.refine_query_system_prompt: str = refine_query_system_prompt

    async def construct_graph_from_chunk(
        self,
        chunk: str,
    ) -> ResponseMessage[list[Triplet]]:
        """Construct graph from a given chunk."""
        messages = self._prepare_messages(chunk)
        result = await call_llm(messages)

        if result is None:
            return self._handle_llm_failure()

        json_result = self._extract_and_repair_json(result)
        if isinstance(json_result, ResponseMessage):
            return json_result  # This carries an error message

        return self._parse_triplets(json_result)

    def _prepare_messages(self, chunk: str) -> list[dict[str, str]]:
        """Prepare system and user messages for LLM call."""
        return [
            {"role": "system", "content": self.graph_system_prompt},
            {"role": "user", "content": f"This is the passage:\n{chunk}"},
        ]

    def _handle_llm_failure(self) -> ResponseMessage[list[Triplet]]:
        """Return a standardized failure response when LLM inference fails."""
        return ResponseMessage[list[Triplet]](error="LLM inference failed")

    def _extract_and_repair_json(self, result: str) -> dict | ResponseMessage:
        """Extract JSON from LLM result and attempt to repair it."""
        json_start, json_end = result.find("{"), result.rfind("}") + 1

        if -1 in (json_start, json_end):
            return ResponseMessage[list[Triplet]](
                error=(
                    "No data from LLM, expect a JSON returned. "
                    f"Received: {result}"
                ),
            )

        try:
            return json_repair.repair_json(
                result[json_start:json_end],
                return_objects=True)
        except json.JSONDecodeError:
            return ResponseMessage[
                list[Triplet]
            ](error=f"Broken JSON generated: {result}")

    def _parse_triplets(
        self,
        json_result: dict,
    ) -> ResponseMessage[list[Triplet]]:
        """Parse and validate triplets from repaired JSON."""
        if isinstance(json_result, list):
            json_result = self._merge_triplets_from_list(json_result)

        if "triplets" not in json_result:
            return ResponseMessage[list[Triplet]](
                error=f"Wrong format of generated JSON: {json_result}",
            )

        resp: list[Triplet] = []
        for item in json_result["triplets"]:
            try:
                triplet = Triplet.model_validate(item)
                resp.append(triplet)
            except Exception:
                logger.exception(f"Failed to validate triplet: {item}")

        return ResponseMessage[list[Triplet]](result=resp)

    def _merge_triplets_from_list(self, items: list) -> dict:
        """Merge triplets from a list of items into a single structure."""
        merged = {"triplets": []}

        for item in items:
            if isinstance(item, dict):
                triplets = item.get("triplets", [])
                if isinstance(triplets, list):
                    merged["triplets"].extend(triplets)

        return merged

    async def refine_query(self, query: str) -> ResponseMessage[str]:
        """Refine query."""
        messages = [
            {
                "role": "system",
                "content": self.refine_query_system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        result = await call_llm_prioritized(messages)
        if result is None:
            return ResponseMessage[str](
                error="LLM inference failed",
            )

        json_start, json_end = result.find("{"), result.rfind("}") + 1

        if -1 in (json_start, json_end):
            return ResponseMessage[str](
                error="No data from LLM, expect a JSON returned",
            )

        try:
            json_result = json_repair.repair_json(
                result[json_start:json_end],
                return_objects=True,
            )
        except json.JSONDecodeError:
            return ResponseMessage[str](
                error="Broken JSON generated",
            )

        return ResponseMessage[str](
            result=str(json_result["refined_query"]),
        )

    async def extract_named_entities(
        self,
        text: str,
    ) -> ResponseMessage[list[str]]:
        """Extract named entities from a given text."""
        messages = [
            {
                "role": "system",
                "content": self.ner_system_prompt,
            },
            {
                "role": "user",
                "content": f"This is the passage:\n{text}",
            },
        ]

        result = await call_llm_prioritized(messages)

        if result is None:
            return ResponseMessage[list[str]](
                error="LLM inference failed",
            )

        json_start, json_end = result.find("{"), result.rfind("}") + 1

        if -1 in (json_start, json_end):
            return ResponseMessage[list[str]](
                error="No data from LLM, expect a JSON returned",
            )

        try:
            json_result = json_repair.repair_json(
                result[json_start:json_end],
                return_objects=True,
            )
        except json.JSONDecodeError:
            return ResponseMessage[list[str]](
                error="Broken JSON generated",
                result=[],
            )

        return ResponseMessage[list[str]](
            result=json_result["entities"],
        )


@lru_cache(maxsize=1)
def get_graph_knowledge() -> GraphKnowledge:
    """Get a GraphKnowledge instance."""
    return GraphKnowledge()
