# How to launch Claude Code from PowerShell

This guide shows you how to start Claude Code sessions against an Ollama model from PowerShell using `scripts/cco.ps1`, including how to pass Claude Code's own flags through to the session.

`cco` picks the bridge for you. It reads the target model's capabilities from `ollama show`, sends a model that reports `vision` straight through `ollama launch claude`, and routes one that does not through `ovp launch`. You do not have to decide per model.

## Before you start

- Ollama running and reachable, `ollama serve`.
- `claude` on `PATH`. A `.cmd` or `.bat` shim is fine; `ovp` runs those through `cmd /c` for you.
- `ovp` on `PATH`, needed only for text-only models. Check with `Get-Command ovp`. If you installed into a virtual environment rather than with `uv tool install`, `ovp` lands at `.venv\Scripts\ovp.exe` and will not be on `PATH` until you put it there. See [SETUP.md](../../SETUP.md).
- A target model that reports the `tools` capability, and one vision-capable model pulled for transcription (`gemma3:4b` unless you override it).

## Load the command

Dot-source the script. Running it without the leading dot defines the function inside the script's own scope and then throws it away, which looks like nothing happened.

```powershell
. C:\path\to\ollama-vision-proxy\scripts\cco.ps1
```

To have it in every session, add that same line to your profile:

```powershell
Add-Content -Path $PROFILE -Value '. C:\path\to\ollama-vision-proxy\scripts\cco.ps1'
```

If the dot-source is refused because of the execution policy, unblock the single file rather than loosening the policy machine-wide:

```powershell
Unblock-File C:\path\to\ollama-vision-proxy\scripts\cco.ps1
```

## Launch a session

Name the model and nothing else:

```powershell
cco glm-5.2:cloud
```

`cco` prints the route it chose before Claude Code takes the screen, so you can confirm what happened:

```
cco: glm-5.2:cloud [thinking,completion,tools] -> via ovp, images transcribed locally
ovp: proxy listening on http://127.0.0.1:59331, forwarding to http://127.0.0.1:11434
```

A vision-capable model reports the other route instead, and no proxy starts:

```powershell
cco qwen3-vl:4b
```

```
cco: qwen3-vl:4b [completion,vision,tools,thinking] -> direct, no vision bridge
```

To force the bridge on a model that has vision, or off on one that does not, pass `--bridge` or `--no-bridge`. Forcing it off means pasted images are rejected by the model.

## Pass Claude Code's own flags

Put a bare `--` after the model. Everything following it goes to `claude` untouched:

```powershell
cco glm-5.2:cloud -- --agent manager
cco glm-5.2:cloud -- --dangerously-skip-permissions --agent secretary
```

Quote any argument containing spaces, as you would for any native command. The tokens reach `claude` intact:

```powershell
cco glm-5.2:cloud -- -p "summarise this repository"
```

The `--` is optional for flags that `cco` does not own, so this is equivalent to the first example:

```powershell
cco glm-5.2:cloud --agent manager
```

Use the `--` form when the flag you want to pass is one `cco` recognises for itself: `--bridge`, `--no-bridge`, `--no-vision`, `--vision-model`, `--proxy-port`, `-h`, and `--help`. Before the `--` those are consumed by `cco`; after it they are passed straight to `claude`. When in doubt, use `--`, since it is never wrong.

## Run several sessions at once

Open another terminal and run `cco` again. Nothing else is needed: each session gets its own proxy on a port the OS picks, so concurrent sessions do not contend for one.

Pin a port only when something else has to reach the proxy. Two sessions given the same pinned port will collide and the second will fail to start.

```powershell
cco glm-5.2:cloud --proxy-port 11500 -- --agent manager
```

## Choose a different transcription model

```powershell
cco glm-5.2:cloud --vision-model minicpm-v
```

The model must report `vision`. Check yours with `ollama show <model>`.

## Verify it worked

Paste an image into the session and ask what it shows. On the bridged route the model answers from a text description rather than rejecting the request, and the reply refers to content it could only have received as transcribed text. If you want the transcription on the record, relaunch with a log file through `ovp` directly:

```powershell
ovp launch --target-model glm-5.2:cloud --log-file $env:TEMP\ovp.log -- --agent manager
```

## If something fails

| What you see | What to do |
|---|---|
| `cco: no model given. Try: cco --help` | Pass the model as the first argument. |
| `cco: ollama is not on PATH` | Install Ollama, or open a new terminal so `PATH` is picked up. |
| `cco: ollama does not know '<model>'` | Check the name with `ollama ls`, then `ollama pull <model>`. |
| `cco: '<model>' reports [...] with no 'tools'` | Claude Code cannot drive that model. Pick one whose capabilities include `tools`. |
| `cco: '<model>' has no vision and ovp is not installed` | Install `ovp`, or add its `.venv\Scripts` directory to `PATH`, or accept no image support with `--no-bridge`. |
| `cco: --vision-model needs a model name` | The flag was last on the line. Give it a value. |
| `Cannot reach the Ollama server at <url>` | Start it with `ollama serve`. |
| `Could not find the claude CLI on PATH` | Install Claude Code, or add it to `PATH`. |
| `Cannot listen on port <n>` | You pinned `--proxy-port` and something holds that port. Drop the flag. |
| Nothing at all happens after loading the file | You ran the script instead of dot-sourcing it. Re-read [Load the command](#load-the-command). |

Windows has never been exercised on a real Windows machine, in this script or in `ovp` itself. If something breaks in a way this table does not cover, treat it as a bug here rather than a mistake of yours.

## Related

- [Options](../../README.md#options) for the full `ovp launch` flag reference.
- [Known limits](../../README.md#known-limits) for what to expect before relying on this.
