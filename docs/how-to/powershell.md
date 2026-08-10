# Running ovp from PowerShell on Windows

This is the Windows companion to [SETUP.md](../../SETUP.md). It covers the parts
that are different on this platform, and the one thing you probably want at the
end of it: a `cso` command that opens Claude Code against an Ollama model with
image support, sitting beside your existing `cs` without disturbing it.

Everything below was executed on Windows PowerShell 5.1 against Ollama 0.32.6,
with `glm-5.2:cloud` as the target model and `gemma3:4b` for vision.

## Install

Windows has no `python3`, and `pip install` into the Microsoft Store Python is a
bad idea for the same reason it is on Debian. Use the launcher and a virtual
environment:

```powershell
git clone https://github.com/paulocfjunior/ollama-vision-proxy.git
cd ollama-vision-proxy
py -m venv .venv
.venv\Scripts\python -m pip install .
```

`ovp` lands at `.venv\Scripts\ovp.exe`. Verify it:

```powershell
.venv\Scripts\ovp --version
```

Expect `ovp 0.1.0`. Putting `.venv\Scripts` on `PATH` is optional: the wrapper in
the next section finds that path on its own.

## Wire up `cso`

[`scripts/cco.ps1`](../../scripts/cco.ps1) defines `cso`, plus the long form
`Invoke-ClaudeOllama` for when a default needs changing. Dot-source it from your
profile:

```powershell
notepad $PROFILE
```

Add one line, with the path to your clone:

```powershell
. "$HOME\ollama-vision-proxy\scripts\cco.ps1"
```

Open a new terminal and run `cso`. The first launch checks Ollama, checks both
models, and offers to pull anything missing.

The defaults live in one array at the top of the script, so change them there
rather than at the call site. It ships with `--dangerously-skip-permissions`, on
the view that a local model is not worth approving tool by tool; if that is not
your view, remove that one line.

If dot-sourcing is refused with `running scripts is disabled on this system`,
your execution policy is `Restricted`. Either relax it for your own account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or, if the clone came from a browser download rather than `git clone`, clear the
mark of the web on the file: `Unblock-File scripts\cco.ps1`.

## Why `cs` still works

`cso` never assigns `$env:ANTHROPIC_BASE_URL`. An assignment in a PowerShell
session outlives the command that made it, so the next `cs` in that window would
quietly go to Ollama instead of Anthropic, and nothing in its output would say
so. Instead `ovp` builds the variable into the environment of the `claude`
process it spawns, and that copy dies with the process.

The practical consequence: `cs` and `cso` can be run in the same window, in
either order, as often as you like.

## Several sessions at once

Each `cso` asks the OS for a free port and tells its own `claude` about it, so
sessions do not collide. Three run concurrently in three windows land on three
consecutive ephemeral ports and answer independently.

They share the Ollama server and therefore the resident models, which is the
point: a second session costs another small proxy process, not another copy of
`gemma3:4b`.

`Invoke-ClaudeOllama -ProxyPort 11500` opts out of this. Two sessions pinned to
one port will collide, and the second will refuse to start.

## Verify the image path

The check from SETUP.md, in PowerShell dialect. Write a test image:

```powershell
py -c "import base64,pathlib,os;pathlib.Path(os.environ['TEMP']+r'\ovp-red.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsEty/UMZxgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGBywLPLIEA68ZURwAAAABJRU5ErkJggg=='))"
```

Then run one headless turn through the proxy, with tracing on so the log records
what happened:

```powershell
$log = "$env:TEMP\ovp-setup.log"
Invoke-ClaudeOllama -Trace -LogFile $log -ClaudeArgs @(
    '-p', "Use the Read tool on $env:TEMP\ovp-red.png, then state in one short sentence what the image shows."
)
```

Judge it by the log, in this order:

```powershell
(Select-String -Path $log -Pattern 'POST /v1/messages').Count
Select-String -Path $log -Pattern 'classified image as|transcription failed'
Select-String -Path $log -Pattern 'transcribed'
```

Expect a count of 1 or more, a classification line with a non-empty reply and no
failure line, and `transcribed 1 image(s), skipped 0`. The reply itself is
corroboration rather than proof; [SETUP.md](../../SETUP.md) explains why at
length.

`-Trace` matters for the first two. Without it the log is at INFO, where the
classification and request lines are not written, and only the `transcribed`
line appears.

One startup line is expected and is not a fault:

```
claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source is set
```

`ovp` blanks `ANTHROPIC_API_KEY` in the child so an inherited real key cannot be
used by accident, and Claude Code notices the blank.

## Windows specifics worth knowing

**A space in the path to `claude` used to break the launch.** npm installs
`claude.cmd`, and CreateProcess cannot execute a `.cmd`, so it has to go through
`cmd.exe`. `cmd /c` strips the first and last quote of its command line whenever
that line begins with a quote, which turns
`cmd /c "C:\Program Files\npm\claude.cmd" -p x` into an attempt to run
`C:\Program`. Starting the line with `call` keeps the quotes intact and still
propagates the exit code. Anyone whose profile directory or install location
contains a space met this on their first launch; it is fixed, and
`test_windows_shim_line_never_begins_with_a_quote` keeps it fixed.

**`cmd` still expands `%VAR%` inside your arguments.** Going through the shim is
unavoidable for a `.cmd`, and a `-p` prompt containing something like
`%USERNAME%` will reach `claude` with the value substituted. A lone `%`, as in
`50% off`, is passed through untouched. There is no `cmd` switch that disables
this; if it matters, put the text in a file and ask Claude Code to read it.

**A bare `--` survives PowerShell.** For native commands, both Windows
PowerShell 5.1 and PowerShell 7 pass `--` through to the process rather than
consuming it, so `ovp launch --target-model glm-5.2:cloud -- --agent x` works as
written. `cco.ps1` does not rely on that anyway: it builds one array and splats
it, so each element becomes its own argv entry regardless of parser mode.

**`cso -p "..."` is forwarded, not bound.** `cso` hands `$args` to
`Invoke-ClaudeOllama` as an explicit `-ClaudeArgs` array. Splatting instead would
let PowerShell try to bind a claude flag such as `-p` to a parameter of the
wrapper, and `-p` is ambiguous against `-ProxyPort`.

**Console output is not UTF-8.** Log files are written as UTF-8 explicitly, but
the terminal handler uses the console code page. Python writes stderr with
`backslashreplace`, so an unrepresentable character is mangled rather than
fatal. Read `-LogFile` rather than the scrollback when a description looks wrong.

## If something fails

| What you see | What it means |
|---|---|
| `running scripts is disabled on this system` | Execution policy. `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or `Unblock-File` the script. |
| `ovp: the ovp command was not found` | The install did not complete, or it landed somewhere the wrapper does not look. Re-run the install and check `.venv\Scripts\ovp.exe` exists. |
| `ovp: cannot reach the Ollama server` | `ollama serve` is not running. |
| `'<model>' is available but does not support 'tools'` | That model cannot drive Claude Code. Pick one whose capabilities include `tools`. |
| `'<model>' is available but does not support 'vision'` | Wrong model for `-VisionModel`. |
| `'C:\Users\...' is not recognized as an internal or external command` | The pre-`call` shim bug, on a build from before this fix. Pull the latest and reinstall. |
| `Cannot listen on port <n>` | You pinned `-ProxyPort` and something else holds it. Drop the flag. |
