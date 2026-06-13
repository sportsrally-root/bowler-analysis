"""Pluggable vision-LLM client for batter shot analysis.

Four backends, one interface (``analyze_images`` -> ``BatterShotAnalysis``):

* ``aws`` / ``bedrock`` / ``anthropic`` — all use the official ``anthropic`` SDK
  (different client constructor + model-id prefix), with native structured output
  via ``messages.parse`` and base64 ``image`` blocks.
* ``databricks`` — an OpenAI-compatible serving endpoint via the ``openai`` SDK,
  with ``image_url`` blocks and ``response_format`` json_schema.

Every backend is defensive: if native structured output fails (older endpoint,
unsupported param), we retry asking for raw JSON and validate it with pydantic.
That covers the "endpoint capabilities unknown" case for Databricks and any
serving setup that doesn't honour ``output_format``/``response_format``.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from ..config import Llm
from ..models.batter_schemas import BatterShotAnalysis

_JSON_INSTRUCTION = (
    "\n\nReturn ONLY a single JSON object that conforms to this JSON Schema. "
    "No prose, no markdown code fences:\n"
)


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model response (tolerates fences/prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
    return text[start:end + 1]


class LlmClient(Protocol):
    def analyze_images(
        self, images_b64: list[str], system: str, user_text: str
    ) -> BatterShotAnalysis: ...


class _AnthropicFamilyClient:
    """aws / bedrock / anthropic — the official anthropic SDK."""

    def __init__(self, cfg: Llm):
        self.cfg = cfg
        if cfg.backend == "aws":
            from anthropic import AnthropicAWS
            self.client = AnthropicAWS()
            self.model = cfg.model
        elif cfg.backend == "bedrock":
            from anthropic import AnthropicBedrock
            self.client = AnthropicBedrock()
            # On-demand Claude on Bedrock needs an inference-profile id
            # (e.g. 'us.anthropic.claude-opus-4-8'). Pass any id that already
            # names the provider through unchanged; only bare ids get prefixed.
            self.model = cfg.model if "anthropic." in cfg.model \
                else f"anthropic.{cfg.model}"
        else:  # anthropic
            from anthropic import Anthropic
            self.client = Anthropic()
            self.model = cfg.model

    def _content(self, images_b64: list[str], user_text: str) -> list[dict]:
        blocks: list[dict] = [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": b}}
            for b in images_b64
        ]
        blocks.append({"type": "text", "text": user_text})
        return blocks

    def analyze_images(self, images_b64, system, user_text) -> BatterShotAnalysis:
        messages = [{"role": "user", "content": self._content(images_b64, user_text)}]
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=self.cfg.max_tokens,
                thinking={"type": "adaptive"},
                system=system,
                messages=messages,
                output_format=BatterShotAnalysis,
            )
            if resp.parsed_output is not None:
                return resp.parsed_output
            raise ValueError("parsed_output was None")
        except Exception:
            # Defensive fallback: ask for raw JSON, validate ourselves.
            schema = json.dumps(BatterShotAnalysis.model_json_schema())
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=[{"role": "user", "content": self._content(
                    images_b64, user_text + _JSON_INSTRUCTION + schema)}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return BatterShotAnalysis.model_validate_json(_extract_json(text))


class _DatabricksClient:
    """OpenAI-compatible Databricks serving endpoint via the openai SDK."""

    def __init__(self, cfg: Llm):
        from openai import OpenAI
        self.cfg = cfg
        host = (cfg.databricks_host or os.environ.get("DATABRICKS_HOST", "")).rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN")
        if not host or not token:
            raise RuntimeError(
                "Databricks backend needs DATABRICKS_HOST (or config.llm.databricks_host) "
                "and DATABRICKS_TOKEN environment variables.")
        self.model = cfg.databricks_endpoint or cfg.model
        self.client = OpenAI(base_url=f"{host}/serving-endpoints", api_key=token)

    def _messages(self, images_b64, system, user_text) -> list[dict]:
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                   for b in images_b64]
        content.append({"type": "text", "text": user_text})
        return [{"role": "system", "content": system},
                {"role": "user", "content": content}]

    def analyze_images(self, images_b64, system, user_text) -> BatterShotAnalysis:
        schema = BatterShotAnalysis.model_json_schema()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.max_tokens,
                messages=self._messages(images_b64, system, user_text),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "batter_shot_analysis",
                                    "schema": schema, "strict": True},
                },
            )
            return BatterShotAnalysis.model_validate_json(
                resp.choices[0].message.content)
        except Exception:
            # Defensive fallback: plain completion asking for raw JSON.
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.max_tokens,
                messages=self._messages(
                    images_b64, system,
                    user_text + _JSON_INSTRUCTION + json.dumps(schema)),
            )
            return BatterShotAnalysis.model_validate_json(
                _extract_json(resp.choices[0].message.content))


class _NovaBedrockClient:
    """Amazon Nova on Bedrock via the boto3 ``converse`` API.

    Nova is Amazon's own (first-party) multimodal model, so it isn't gated behind
    the third-party Marketplace subscription that Anthropic-on-Bedrock requires —
    it works on a plain Bedrock-enabled account. No native structured output, so
    we always ask for JSON and validate with pydantic.
    """

    def __init__(self, cfg: Llm):
        import boto3
        self.cfg = cfg
        region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                  or "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)
        m = cfg.model if "nova" in cfg.model else "us.amazon.nova-pro-v1:0"
        if m.startswith("amazon."):       # on-demand needs an inference profile id
            m = "us." + m
        self.model = m

    def analyze_images(self, images_b64, system, user_text) -> BatterShotAnalysis:
        import base64
        schema = json.dumps(BatterShotAnalysis.model_json_schema())
        content = [{"image": {"format": "jpeg",
                              "source": {"bytes": base64.b64decode(b)}}}
                   for b in images_b64]
        content.append({"text": user_text + _JSON_INSTRUCTION + schema})
        resp = self.client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": self.cfg.max_tokens},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return BatterShotAnalysis.model_validate_json(_extract_json(text))


def make_client(cfg: Llm) -> LlmClient:
    """Construct the configured backend client."""
    backend = cfg.backend.lower()
    if backend in ("aws", "bedrock", "anthropic"):
        return _AnthropicFamilyClient(cfg)
    if backend == "nova":
        return _NovaBedrockClient(cfg)
    if backend == "databricks":
        return _DatabricksClient(cfg)
    raise ValueError(f"Unknown llm.backend: {cfg.backend!r} "
                     "(expected aws | bedrock | nova | anthropic | databricks)")
