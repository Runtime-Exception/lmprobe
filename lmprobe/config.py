"""Config parser for api.txt format."""

import re
from pathlib import Path
from dataclasses import dataclass
from .models import Endpoint


@dataclass
class ConfigResult:
    """Result of parsing a config file."""
    endpoints: list[Endpoint]
    is_probed: bool = False
    probed_timestamp: int | None = None


def parse_api_config(file_path: str | Path) -> list[Endpoint]:
    """Parse the api.txt file and extract endpoints, keys, and models.

    Format:
    - Line starting with http:// or https:// = base URL
    - Lines starting with sk- = API keys
    - Lines with commas = model list (comma-separated)
    - Empty lines = separator between endpoint blocks
    """
    return parse_api_config_ex(file_path).endpoints


def parse_api_config_ex(file_path: str | Path) -> ConfigResult:
    """Parse the api.txt file with extended metadata.

    Returns ConfigResult with endpoints and probed status.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    endpoints: list[Endpoint] = []
    current_url: str | None = None
    current_keys: list[str] = []
    current_models: list[str] = []
    is_probed = False
    probed_timestamp = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Check for PROBED marker
            if line.startswith("#PROBED:"):
                try:
                    probed_timestamp = int(line.split(":")[1])
                    is_probed = True
                except (ValueError, IndexError):
                    is_probed = True
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            # Skip empty lines - finalize current endpoint if we have one
            if not line:
                if current_url and current_keys:
                    endpoints.append(Endpoint(
                        base_url=current_url,
                        api_keys=current_keys,
                        claimed_models=current_models
                    ))
                current_url = None
                current_keys = []
                current_models = []
                continue

            # Check for URL
            if line.startswith("http://") or line.startswith("https://"):
                # Save previous endpoint if exists
                if current_url and current_keys:
                    endpoints.append(Endpoint(
                        base_url=current_url,
                        api_keys=current_keys,
                        claimed_models=current_models
                    ))
                current_url = line.rstrip("/")
                current_keys = []
                current_models = []
                continue

            # Check for API key (sk- prefix is common)
            if line.startswith("sk-") or re.match(r"^[a-zA-Z0-9_-]{20,}$", line):
                current_keys.append(line)
                continue

            # Check for model list (contains commas)
            if "," in line:
                models = [m.strip() for m in line.split(",") if m.strip()]
                current_models.extend(models)
                continue

            # Single model without comma
            if current_url:
                current_models.append(line)

    # Don't forget the last endpoint
    if current_url and current_keys:
        endpoints.append(Endpoint(
            base_url=current_url,
            api_keys=current_keys,
            claimed_models=current_models
        ))

    return ConfigResult(
        endpoints=endpoints,
        is_probed=is_probed,
        probed_timestamp=probed_timestamp
    )


def get_all_claimed_models(endpoints: list[Endpoint]) -> dict[str, list[tuple[str, str]]]:
    """Get a mapping of model_id -> list of (endpoint_url, api_key) that claim to support it."""
    model_providers: dict[str, list[tuple[str, str]]] = {}

    for endpoint in endpoints:
        for model in endpoint.claimed_models:
            if model not in model_providers:
                model_providers[model] = []
            for key in endpoint.api_keys:
                model_providers[model].append((endpoint.base_url, key))

    return model_providers


def get_unique_models(endpoints: list[Endpoint]) -> list[str]:
    """Get sorted list of unique model IDs across all endpoints."""
    models = set()
    for endpoint in endpoints:
        models.update(endpoint.claimed_models)
    return sorted(models)
