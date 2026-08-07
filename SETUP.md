# How to wire Claude Code to ollama-vision-proxy

This runbook takes a fresh clone of this repo to a working install, verifies the whole image path without needing an interactive session, and ends with the one command your human has to run themselves.

It is written to be executed by a Claude Code agent. Every step has a command and an observable result, so no step is ever "probably fine". If a result does not appear as described, stop at that step and say so rather than continuing.

If you are a human reading this, you can follow it too, but [README.md](README.md) is the friendlier door.

## The one step you cannot do for yourself

**You cannot point your own session at the proxy.** `ANTHROPIC_BASE_URL` is read by the `claude` process when it starts, and your process has already started. Nothing you run in this session changes where your own requests go.

So the finish line for this runbook is not "the agent is now using the proxy". It is: the proxy is installed, proven to work, and the human has a command to run in a new terminal. Do not try to work around this by editing their shell profile, exporting variables, or restarting anything. Just hand over the command at the end.

## Before you start

- Ollama installed, with `ollama serve` running.
- The `claude` CLI on `PATH`.
- Python 3.9 or newer.
- Roughly 3 to 4 GB of free disk if a vision model still has to be pulled. **Ask your human before pulling one.** It is their disk and their bandwidth.

## Step 1. Confirm Ollama is reachable

```bash
curl -s http://127.0.0.1:11434/api/version
```

Expect a version object, for example `{"version":"0.32.5"}`. Empty output means the server is not running: ask your human to start it with `ollama serve`.

On Windows PowerShell, use `Invoke-RestMethod http://127.0.0.1:11434/api/version` instead.

## Step 2. Confirm the claude CLI is on PATH

```bash
which claude    # Windows PowerShell: Get-Command claude
```

Expect a path. If there is none, `ovp` refuses to start later, reporting that it could not find the `claude` CLI on `PATH`.

## Step 3. Choose the target model, and check it supports tools

The target model is the one Claude Code talks to. It has to satisfy two conditions, and only one of them is obvious.

```bash
ollama show <candidate-model>
```

Read the `Capabilities` block and decide:

| Capabilities show | Verdict |
|---|---|
| `tools` present, `vision` absent | Use it. This is exactly what the proxy exists for. |
| `tools` absent | Unusable. Claude Code needs tool support, whatever else the model can do. |
| `tools` and `vision` both present | It works, but it can already see images, so you do not need this proxy for them. |

The `tools` requirement is easy to miss because nothing warns you up front. A model without it gets as far as the first turn and then fails inside Claude Code:

```
API Error: 400 registry.ollama.ai/library/gemma3:1b does not support tools
```

`ovp` does not preflight the target model at all, so a name that is simply wrong also survives until the first turn, and then returns `HTTP 404 model "<name>" not found`. Confirm the name with `ollama show` now and you avoid both.

If your human has no suitable model, `gemma4:e2b-mlx` reports `completion`, `tools` and `thinking`, with no `vision`, which is the right shape. It is about 6.5 GB, so ask before pulling it. The README's own example is `glm-5.2:cloud`, which Ollama serves from its cloud rather than from local weights.

## Step 4. Choose the vision model, and check it supports vision

This is the local model that describes images. The default is `gemma3:4b`.

```bash
ollama show gemma3:4b
```

Expect `vision` in the `Capabilities` block. If the model is not pulled at all, `ovp launch` offers to pull it for you and `-y` accepts without asking, but ask your human before you use `-y`.

Note that the two roles do not overlap: `gemma3:4b` has `vision` and no `tools`, so it makes a good vision model and cannot be a target model. Picking one model for both jobs usually satisfies neither.

## Step 5. Install ovp

Use an isolated install. Pick the first of these that is available:

```bash
uv tool install .        # then: uv tool update-shell, if ovp is not on PATH
pipx install .           # then: pipx ensurepath, if ovp is not on PATH
```

If neither `uv` nor `pipx` is present, use a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
```

On Windows, that is `py -m venv .venv` and `.venv\Scripts\python -m pip install .`, and `ovp` lives at `.venv\Scripts\ovp.exe`.

**Do not run `pip install` outside a virtual environment.** On a Homebrew or Debian Python it stops with:

```
error: externally-managed-environment
```

That is PEP 668 protecting the system interpreter, and it is not a problem with this package.

Verify the install:

```bash
ovp --version
```

Expect `ovp 0.1.0`. If the command is not found, the install worked but its `bin` directory is not on `PATH`: run the shell command listed beside your install method above, or call the executable by its full path for the rest of this runbook.

## Step 6. Verify the whole chain, without an interactive session

This is the step that proves the proxy actually does its job. Write a test image, then send Claude Code at it through the proxy in headless mode. On Windows, substitute a writable path such as `$env:TEMP\ovp-red.png` for `/tmp/ovp-red.png` throughout this step.

```bash
python3 -c "import base64,pathlib;pathlib.Path('/tmp/ovp-red.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsEty/UMZxgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGBywLPLIEA68ZURwAAAABJRU5ErkJggg=='))"
```

That writes a 64x64 solid red PNG. Now run it through the proxy in verbose mode, substituting your target model from step 3, so the log records what the proxy actually did:

```bash
ovp launch --target-model <target-model> -v --log-file /tmp/ovp-setup.log \
  -- -p "Use the Read tool on /tmp/ovp-red.png, then state in one short sentence what the image shows."
```

It takes a minute or two the first time, because both models load cold. Verbose mode also spills the log onto the terminal; ignore that and read the file.

**Judge the run by the log, in this order.** What the model said is corroboration, not proof, for reasons below. On Windows, use `Select-String` in place of `grep`.

**Check 1. Claude Code went through the proxy at all.**

```bash
grep -c 'POST /v1/messages' /tmp/ovp-setup.log
```

Expect 1 or more. A zero means `ANTHROPIC_BASE_URL` never reached the child process, and nothing below it means anything.

**Check 2. The vision model actually answered, rather than failing quietly.**

```bash
grep -E "classified image as|transcription failed" /tmp/ovp-setup.log
```

Expect a classification line whose reply is not empty, and no failure line:

```
DEBUG ollama_vision_proxy.vision: classified image as other (reply 'OTHER')
```

An empty reply there means the vision model returned nothing and every image is quietly falling through to the generic path, which reads downstream exactly like a working setup.

**Check 3. The image was replaced.**

```bash
grep transcribed /tmp/ovp-setup.log
```

Expect `transcribed 1 image(s), skipped 0`. Note that this count includes failed transcriptions, which is why check 2 comes first: on its own, this line cannot tell a real description from a fail-soft placeholder.

### Why the model's reply is not the test

All three of these were observed on the same machine within minutes, on a setup that was correctly wired the whole time:

- `The image shows a solid block of vibrant red color filling the entire frame.` The good case.
- `I am waiting for your next instruction.` A clean log, and a target model that simply did not follow the instruction. The setup is fine.
- `The image shows a schematic diagram of a simple REST API flow...` for a plain red square. Here the model never called `Read`, so no image ever entered the request, no `transcribed` line appeared, and it invented an answer instead. Also not a proxy fault.

So if checks 1 to 3 pass, the setup is proven regardless of the prose. If check 3 is missing, look at the reply before blaming the wiring: a model that never called `Read` produces no image to transcribe. Retry once, or use a stronger target model.

One inverse case is worth catching: a reply that describes the image correctly while check 3 shows **nothing** means the target model read the image itself, so it has `vision` and step 3 was answered wrong.

One line of startup output is expected and is not a fault:

```
⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source is set
```

`ovp` deliberately blanks `ANTHROPIC_API_KEY` in the child process so an inherited real key cannot be used by accident. Claude Code notices the blank and says so.

## Step 7. Hand the command to your human

Report the verified setup and give them this to run in a new terminal, with the target model filled in:

```bash
ovp launch --target-model <target-model>
```

Tell them three things:

- Use `ovp launch` wherever they would have used `ollama launch claude`.
- Anything after a bare `--` is passed to `claude` untouched, so `ovp launch --target-model <target-model> -- --agent manager` works.
- Each image costs one vision call, around 20 seconds for a screenshot with the default model, and the first one also pays the model's cold load. An image already sitting in the conversation history is not described again.

Then stop. Do not attempt to redirect your own session.

## If a step fails

| What you see | What it means |
|---|---|
| `Cannot reach the Ollama server at <url>` | `ollama serve` is not running, or `--upstream-url` points at the wrong place. |
| `The model '<x>' is pulled but does not support vision` | Wrong model in `--vision-model`. The message lists the vision-capable models already pulled. |
| `The vision model '<x>' is not available in Ollama` | Not pulled. `ollama pull <x>`, or let `ovp launch` offer. |
| `Cannot listen on port 11435` | Another process holds the port. Pass `--proxy-port`. |
| Could not find the `claude` CLI on PATH | Claude Code is not installed, or not on `PATH`. |
| `does not support tools` | The target model cannot drive Claude Code. Go back to step 3. |
| `404 model "<x>" not found` | The target model name is wrong or not pulled. Check with `ollama show <x>`. |
| `error: externally-managed-environment` | `pip install` was run outside a virtual environment. Go back to step 5. |
| `transcription failed (...)` inside an `<image-transcription>` block | The session is fine and the request went through; only that one description failed. Raise `--vision-timeout`, or check the log. |

A note on Windows: the code paths for it are written and unit-tested, but have never been executed on Windows by anyone. If you are the first, treat an unexpected failure there as a likely bug in this repo rather than a mistake of yours.

## Where to look next

- [README.md](README.md) for the full flag reference, the vision-model comparison, and what the proxy does to a request.
- The `Options` table in the README for everything not used here, including `--no-geocode`, `--vision-context`, and `--proxy-port`.
