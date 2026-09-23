"""Small Amazon Bedrock Converse adapter for JevAny planning examples."""


class BedrockGenerator:
    """Generate text through Bedrock without enabling model reasoning."""

    def __init__(self, model: str, *, profile: str | None = None,
                 region: str | None = None, client=None):
        self.model = model
        if client is None:
            try:
                import boto3
            except ImportError as error:
                raise ImportError("install JevAny with the bedrock extra") from error
            session = boto3.Session(profile_name=profile, region_name=region)
            client = session.client("bedrock-runtime")
        self.client = client

    def generate(self, prompt: str, params: dict | None = None) -> dict:
        params = params or {}
        supported = {"max_output_tokens", "temperature", "top_p", "stop_sequences"}
        unknown = set(params) - supported
        if unknown:
            raise ValueError(f"unsupported Bedrock generation parameters: {sorted(unknown)}")
        names = {
            "max_output_tokens": "maxTokens",
            "temperature": "temperature",
            "top_p": "topP",
            "stop_sequences": "stopSequences",
        }
        inference = {names[key]: value for key, value in params.items()}
        request = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
        }
        if inference:
            request["inferenceConfig"] = inference
        response = self.client.converse(**request)
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return {
            "text": "\n".join(block["text"] for block in blocks if "text" in block),
            "usage": response.get("usage"),
            "stop_reason": response.get("stopReason"),
        }
