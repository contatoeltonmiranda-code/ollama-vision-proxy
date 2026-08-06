# ollama-vision-proxy

Give image support to text-only Ollama models. `ovp` sits between Claude Code and your Ollama server, replaces every image you paste with a text description produced by a local vision model, and forwards a text-only request to the model you actually want to talk to.

## The problem it solves

Claude Code sends a pasted image as an Anthropic `image` content block. A text-only model rejects the whole request:

```
HTTP 400
Multimodal data provided, but model does not support multimodal requests.
```

The failure is not a one-off. The image stays in the conversation history, so every following request carries it too and fails the same way, which leaves restarting Claude Code as the only way out.

This is emitted by Ollama's own multimodal check, not by a specific backend, so it affects any text-only Ollama model: a remote one like `glm-5.2:cloud` and a local one like `gemma3:1b` alike.

## How it works

```
Claude Code
    |  POST /v1/messages   (contains image blocks)
    v
ovp proxy  127.0.0.1:11435
    |  POST /api/chat      (local vision model describes each image)
    |<---------------------------------------------
    |  POST /v1/messages   (text-only, images replaced by their descriptions)
    v
Ollama server  127.0.0.1:11434  ->  Ollama Cloud (for :cloud models)
```

Each image block becomes a text block holding the description inside an `<image-transcription>` wrapper, followed by a `<metadata>` block when the image carries a GPS fix. Everything else in the request is left exactly as it was, and the streamed response is relayed straight back to Claude Code chunk by chunk.

## Requirements

- Python 3.9 or newer
- Ollama installed and running (`ollama serve`), reachable on `http://127.0.0.1:11434`
- Claude Code, with `claude` on your `PATH`
- One local vision-capable model (see below)

## Install

Clone the repo, then install it.

**macOS and Linux**

```bash
git clone <repo-url> ollama-vision-proxy
cd ollama-vision-proxy
python3 -m pip install -e .
```

**Windows (PowerShell)**

```powershell
git clone <repo-url> ollama-vision-proxy
cd ollama-vision-proxy
py -m pip install -e .
```

Both platforms run the same code. There are no native components, and the only runtime dependency is `httpx`.

Verify the install:

```bash
ovp --version
```

## Pull a vision model

```bash
ollama pull gemma3:4b
```

`gemma3:4b` is the default: roughly 3.3 GB, and good at general description. If a model is missing, `ovp launch` offers to pull it for you (`-y` pulls without asking).

Other options:

| Model | Size | Notes |
|---|---|---|
| `gemma3:4b` | ~3.3 GB | Default. Solid all-round description. |
| `minicpm-v` | ~5.5 GB | Stronger OCR. Prefer it when your images are mostly screenshots of text, logs, or code. |
| `qwen3.5:4b-mlx` | ~4.0 GB | Vision plus thinking and tools. |

Any model Ollama reports a `vision` capability for will work:

```bash
ollama show gemma3:4b
```

`ovp` checks that capability at startup rather than only checking that the model is present, because a pulled but text-only model (`gemma3:1b`, for instance) would otherwise fail on your first pasted image instead of at launch.

## Usage

Use `ovp launch` where you would have used `ollama launch claude`. Everything after a bare `--` goes to the `claude` CLI untouched.

```bash
# Basic
ovp launch --target-model glm-5.2:cloud

# Pass arguments through to claude
ovp launch --target-model glm-5.2:cloud -- --agent manager

# A different vision model and port
ovp launch --target-model glm-5.2:cloud --vision-model minicpm-v --proxy-port 11500
```

The same commands work verbatim in PowerShell.

`ovp` starts the proxy, spawns `claude` pointed at it, and shuts the proxy down when `claude` exits.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--target-model` | required | The text-only model Claude Code talks to, for example `glm-5.2:cloud`. |
| `--vision-model` | `gemma3:4b` | Local Ollama model used to describe images. |
| `--proxy-port` | `11435` | Port the proxy listens on. |
| `--upstream-url` | `http://127.0.0.1:11434` | The Ollama server to forward to. |
| `--vision-timeout` | `180` | Seconds to wait for one transcription. |
| `--vision-context` | `8192` | Context window for the vision model. Pinned so a large `OLLAMA_CONTEXT_LENGTH` cannot reserve tens of GB per image. |
| `-y`, `--yes` | off | Pull a missing vision model without asking. |
| `--geocode-url` | Nominatim | Reverse geocoding endpoint used to name a photo's location. |
| `--no-geocode` | off | Skip the address lookup; coordinates are still reported. |
| `--log-file` | none | Write full logs to this file instead of the terminal. |
| `-v`, `--verbose` | off | Debug logging, kept on the terminal even during the session. |

### The terminal stays clean

Startup messages are printed before Claude Code takes over the screen, and the cache summary after it exits. In between, nothing is written to the terminal, because Claude Code is drawing its interface there and any stray line lands in the middle of it. Use `--log-file /tmp/ovp.log` to keep the full record, or `-v` when you would rather watch the traffic live and accept the mess.

Failed transcriptions still reach you through the conversation itself, as `transcription failed (reason)` in place of the description, which is the right channel for it.

### Why it does not wrap `ollama launch`

`ollama launch claude` sets `ANTHROPIC_BASE_URL` to the Ollama port itself, which would undo the redirect. So `ovp` spawns `claude` directly and reproduces the same environment, with `ANTHROPIC_BASE_URL` pointing at the proxy and the same `OLLAMA_*` performance flags (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=1`).

## How an image is processed

Two vision calls, because one generic prompt served every image badly. It padded prose with preambles, and on a city skyline it confidently named the wrong buildings.

1. **Classify.** One short call asks what kind of image this is: screenshot, document, diagram, photo, or other.
2. **Describe.** A second call uses the prompt that kind deserves. A screenshot gets verbatim text, application names, and any error or stack trace reported in full. A photograph gets its subjects in detail first, then the background. A diagram gets every label plus the structure connecting them.

Both calls pin the context window with `num_ctx` (default 8192, see `--vision-context`). This matters more than it sounds: on a host running `ollama serve` with `OLLAMA_CONTEXT_LENGTH=262144`, a 1.9 GB model reserved **32 GB** of KV cache for a single image. Pinning the context cut that to 2.5 GB and halved the latency, for byte-identical output. Without it, one pasted image can evict every other model on the machine.

Both calls run at `temperature 0`. The models ship at temperature 1, which made transcription a lottery: the same image produced different text run to run, and one run declared an image had no text when it plainly did. Every prompt also forbids inventing proper nouns that are not written in the image.

If classification fails, the description still happens with the generic prompt.

## The metadata block

When an image carries a GPS fix, a `<metadata>` block follows the description:

```
<metadata>
taken    2026-05-04 14:30:00 -04:00
make     Apple
device   iPhone
lat      43.64250000 N
long     79.38722222 W
alt      76.50 m
road     Bremner Boulevard
zipcode  M5V 2T6
suburb   Entertainment District
state    Ontario
city     Toronto
country  Canada
</metadata>
```

Images without a GPS fix, which includes every screenshot, get no block at all.

The EXIF is read from the image bytes with nothing but the standard library. Claude Code re-encodes what you paste, so a 2.5 MB HEIC from a phone arrives as a much smaller JPEG, but the APP1 EXIF segment survives, which means no filesystem access and no extra dependency.

The address comes from reverse geocoding, and it is the only part of this tool that talks to the internet. Only two rounded coordinates are sent, never the image, and results are cached so a photo sitting in conversation history does not re-query every turn. Use `--no-geocode` to keep the coordinates but skip the lookup, or `--geocode-url` to point at your own Nominatim instance. If the lookup fails the coordinates are still reported.

The block sits **outside** the `<image-transcription>` wrapper on purpose. Inside it, the model is told to disregard what it reads; these are the proxy's own facts, not the vision model's output. Values are sanitised, since EXIF strings and geocoder replies are attacker-influenced.

## Behaviour worth knowing

**Each image is transcribed once.** Descriptions are cached by SHA256 of the image data for the life of the proxy, so an image sitting in conversation history costs one vision call, not one per turn. Concurrent requests for the same image collapse into a single call.

**Transcription never breaks your session.** If the vision model times out, is missing, or returns something unusable, the block becomes `[Image: transcription failed (reason)]` and the request still goes through. Turning a vision hiccup into a failed request would recreate the very bug this tool exists to fix.

**Streaming is preserved.** Server-sent events are relayed as they arrive, so tokens appear while the model is still generating. Nothing is buffered.

**Tool and MCP screenshots are covered too.** Images nested inside `tool_result` blocks are transcribed like any other, not just the ones you paste directly.

**Only `/v1/messages` is touched.** Every other path, and any body that is not JSON, is forwarded byte for byte.

## Scope

- Only the Anthropic Messages API (`POST /v1/messages`) is inspected.
- Only local Ollama vision models are used. No cloud vision APIs.
- Images given as a URL (`source.type: "url"`) are not fetched. They become a short placeholder so the request still succeeds.
- Ollama is the only supported provider.
- Install from a clone. Not published to PyPI.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
.venv/bin/python -m pytest
```

257 tests cover EXIF parsing (including truncated and hostile bytes), reverse geocoding, image-kind prompts, the metadata block, image detection and replacement (including nested `tool_result` images), cache and single-flight behaviour, fail-soft transcription, the launcher environment and signal handling, the CLI lifecycle, and full proxy round trips against a fake upstream, streaming and mid-stream failure included.

## Troubleshooting

**`Cannot reach the Ollama server`** Start it with `ollama serve`, or point `--upstream-url` somewhere else.

**`The model '<x>' is pulled but does not support vision`** That model has no vision capability. The error lists the vision-capable models you already have.

**`Cannot listen on port 11435`** Something else has the port. Pass `--proxy-port`.

**`Could not find the claude CLI on PATH`** Install Claude Code, or add it to `PATH`.

**Images still rejected.** Confirm Claude Code is actually going through the proxy: run with `-v` and check for a `POST /v1/messages` log line on each turn. If nothing appears, `ANTHROPIC_BASE_URL` is not reaching the child process.

**Transcriptions are vague.** Try `--vision-model minicpm-v` for text-heavy screenshots, or a larger vision model.
