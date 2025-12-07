#!/usr/bin/env python3
"""LMProbe - Model availability detection and API proxy server."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from lmprobe.server import LMProbeServer
from lmprobe.config import parse_api_config, get_unique_models
from lmprobe.prober import Prober, ModelRegistry


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="LMProbe - Auto-detect model availability and proxy API requests"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "-c", "--config",
        default="api.txt",
        help="Path to API config file (default: api.txt)"
    )
    serve_parser.add_argument(
        "-H", "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "-i", "--probe-interval",
        type=float,
        default=300.0,
        help="Interval between probes in seconds (default: 300)"
    )
    serve_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    serve_parser.add_argument(
        "--force-probe",
        action="store_true",
        help="Force probing even if config file is marked as pre-probed"
    )

    # Probe command (one-time probe)
    probe_parser = subparsers.add_parser("probe", help="Run one-time probe and show results")
    probe_parser.add_argument(
        "-c", "--config",
        default="api.txt",
        help="Path to API config file (default: api.txt)"
    )
    probe_parser.add_argument(
        "-o", "--output",
        help="Output .txt file with available models (same format as input)"
    )
    probe_parser.add_argument(
        "-m", "--model",
        help="Only probe specific model"
    )
    probe_parser.add_argument(
        "-r", "--regex",
        metavar="PATTERN",
        help="Only probe models matching regex pattern (e.g., 'gpt-4.*', 'claude-3')"
    )
    probe_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    probe_parser.add_argument(
        "-f", "--first",
        type=int,
        metavar="N",
        help="Only probe the first N endpoints"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List configured endpoints and models")
    list_parser.add_argument(
        "-c", "--config",
        default="api.txt",
        help="Path to API config file (default: api.txt)"
    )
    list_parser.add_argument(
        "--endpoints",
        action="store_true",
        help="List endpoints only"
    )
    list_parser.add_argument(
        "--models",
        action="store_true",
        help="List unique models only"
    )

    args = parser.parse_args()

    if args.command == "serve":
        setup_logging(args.verbose)
        run_server(args)
    elif args.command == "probe":
        setup_logging(args.verbose)
        asyncio.run(run_probe(args))
    elif args.command == "list":
        run_list(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_server(args):
    """Start the API server."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    server = LMProbeServer(
        config_path=str(config_path),
        host=args.host,
        port=args.port,
        probe_interval=args.probe_interval,
        force_probe=args.force_probe
    )

    app = server.create_app()

    print(f"Starting LMProbe server on http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Claude-compatible endpoint: http://{args.host}:{args.port}/v1/messages")
    print(f"Model list: http://{args.host}:{args.port}/v1/models")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


async def run_probe(args):
    """Run one-time probe."""
    import json
    import re

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    endpoints = parse_api_config(config_path)

    # Limit to first N endpoints if specified
    if args.first:
        endpoints = endpoints[:args.first]
        print(f"Loaded {len(endpoints)} endpoints (limited to first {args.first})")
    else:
        print(f"Loaded {len(endpoints)} endpoints")

    registry = ModelRegistry()
    prober = Prober(endpoints, registry, max_concurrent=30)

    if args.model:
        # Probe specific model
        results = []
        for endpoint in endpoints:
            if args.model in endpoint.claimed_models:
                for key in endpoint.api_keys[:1]:  # Only first key
                    result = await prober.probe_model(args.model, endpoint.base_url, key)
                    results.append(result)
                    status = "OK" if result.success else f"FAIL: {result.error}"
                    print(f"{endpoint.base_url}: {status}")
    elif args.regex:
        # Probe models matching regex pattern
        try:
            pattern = re.compile(args.regex, re.IGNORECASE)
        except re.error as e:
            print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
            sys.exit(1)

        # Find all matching models across endpoints
        results = []
        tasks = []
        for endpoint in endpoints:
            matching_models = [m for m in endpoint.claimed_models if pattern.search(m)]
            if matching_models:
                print(f"{endpoint.base_url}: {len(matching_models)} matching models")
                for model in matching_models[:100]:  # Limit per endpoint
                    for key in endpoint.api_keys[:1]:
                        tasks.append((model, endpoint.base_url, key))

        if not tasks:
            print(f"No models matching pattern '{args.regex}' found")
            await prober.close()
            return

        print(f"\nProbing {len(tasks)} model/endpoint combinations matching '{args.regex}'...")

        for i, (model, url, key) in enumerate(tasks, 1):
            result = await prober.probe_model(model, url, key)
            results.append(result)
            status = "OK" if result.success else "FAIL"
            print(f"[{i}/{len(tasks)}] {model} @ {url[:40]}: {status}")
    else:
        # Probe all
        total = sum(min(len(e.claimed_models), 100) for e in endpoints)
        print(f"Probing {total} model/endpoint combinations...")

        def progress(completed, total, result):
            status = "OK" if result.success else "FAIL"
            print(f"[{completed}/{total}] {result.model_id} @ {result.endpoint_url[:40]}: {status}")

        results = await prober.probe_all(progress_callback=progress)

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nSummary:")
    print(f"  Total probed: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        print(f"\nAvailable models ({len(set(r.model_id for r in successful))}):")
        models_by_name = {}
        for r in successful:
            if r.model_id not in models_by_name:
                models_by_name[r.model_id] = []
            models_by_name[r.model_id].append(r.endpoint_url)

        for model, eps in sorted(models_by_name.items()):
            print(f"  {model}: {len(eps)} provider(s)")

    if args.output:
        # Generate .txt file in same format as input with only available models
        # Group results by endpoint
        endpoint_results: dict[str, dict] = {}
        for r in successful:
            if r.endpoint_url not in endpoint_results:
                endpoint_results[r.endpoint_url] = {
                    "models": set(),
                    "api_key_prefix": r.api_key_prefix
                }
            endpoint_results[r.endpoint_url]["models"].add(r.model_id)

        # Find full API keys from original endpoints
        endpoint_keys: dict[str, list[str]] = {}
        for ep in endpoints:
            if ep.base_url in endpoint_results:
                # Find keys that match the prefix used
                prefix = endpoint_results[ep.base_url]["api_key_prefix"].rstrip("...")
                matching_keys = [k for k in ep.api_keys if k.startswith(prefix)]
                if matching_keys:
                    endpoint_keys[ep.base_url] = matching_keys
                else:
                    endpoint_keys[ep.base_url] = ep.api_keys[:1]

        # Write output file with PROBED marker
        import time as time_module
        timestamp = int(time_module.time())
        with open(args.output, "w") as f:
            # Write marker indicating this file contains verified models
            f.write(f"#PROBED:{timestamp}\n\n")
            for url, data in endpoint_results.items():
                f.write(f"{url}\n")
                for key in endpoint_keys.get(url, []):
                    f.write(f"{key}\n")
                models_list = ", ".join(sorted(data["models"]))
                f.write(f"{models_list}\n")
                f.write("\n")

        print(f"\nAvailable models saved to {args.output}")
        print(f"  Endpoints: {len(endpoint_results)}")
        print(f"  Models: {sum(len(d['models']) for d in endpoint_results.values())}")
        print(f"  Marked as pre-probed (server will skip initial probe)")

    await prober.close()


def run_list(args):
    """List configured endpoints and models."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    endpoints = parse_api_config(config_path)

    if args.endpoints:
        print("Configured endpoints:")
        for ep in endpoints:
            print(f"  {ep.base_url}")
            print(f"    Keys: {len(ep.api_keys)}")
            print(f"    Models: {len(ep.claimed_models)}")
    elif args.models:
        models = get_unique_models(endpoints)
        print(f"Unique models ({len(models)}):")
        for model in models:
            # Count how many endpoints claim this model
            count = sum(1 for ep in endpoints if model in ep.claimed_models)
            print(f"  {model} ({count} endpoints)")
    else:
        print(f"Config: {config_path}")
        print(f"Endpoints: {len(endpoints)}")
        models = get_unique_models(endpoints)
        print(f"Unique models: {len(models)}")
        total_keys = sum(len(ep.api_keys) for ep in endpoints)
        print(f"Total API keys: {total_keys}")


if __name__ == "__main__":
    main()
