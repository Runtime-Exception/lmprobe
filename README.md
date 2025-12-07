# LMProbe

Auto-detect LLM API model availability and serve as a unified proxy with OpenAI and Claude-compatible endpoints.
可用于打野自动测活

## Features

- **Auto-detection**: Probes API endpoints to verify which models are actually available
- **Unified proxy**: Single API server supporting both OpenAI and Claude formats
- **Pre-probed configs**: Save verified models to skip probing on server start
- **Background polling**: Continuously updates model availability
- **Regex filtering**: Probe specific models matching patterns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Probe endpoints

```bash
# Probe all endpoints
python main.py probe -c api.txt

# Probe first 5 endpoints only
python main.py probe -c api.txt -f 5

# Probe models matching regex
python main.py probe -c api.txt -r "gpt-4.*"

# Save available models to file
python main.py probe -c api.txt -o available.txt
```

### Start server

```bash
# Start with raw config (probes on startup)
python main.py serve -c api.txt

# Start with pre-probed config (skips probing)
python main.py serve -c available.txt

# Force probing even with pre-probed config
python main.py serve -c available.txt --force-probe
```

### List configured endpoints

```bash
python main.py list -c api.txt
python main.py list -c api.txt --models
python main.py list -c api.txt --endpoints
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models (OpenAI format) |
| `POST /v1/chat/completions` | Chat completion (OpenAI format) |
| `POST /v1/messages` | Chat completion (Claude format) |
| `GET /status` | Show all available models |
| `POST /probe` | Trigger manual probe |

## Config Format

```
https://api.example.com
sk-your-api-key-here
gpt-4o, gpt-4o-mini, gpt-4-turbo

https://api.another.com
sk-another-key
claude-3-5-sonnet, claude-3-opus
```

Pre-probed files include a marker:
```
#PROBED:1733567890

https://api.example.com
...
```

## License

MIT
