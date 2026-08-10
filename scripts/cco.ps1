<#
.SYNOPSIS
    Claude Code against any Ollama model, bridging vision only when it is needed.

.DESCRIPTION
    One command that starts a Claude Code session against an Ollama model and
    decides for itself whether the session needs `ovp` in front of it.

    The bridge decision comes from the model's own reported capabilities rather
    than from memory: a model that reports `vision` is launched straight through
    ollama, and one that does not is routed through `ovp`, which transcribes
    pasted images with a local vision model before forwarding. `tools` is
    required either way, since Claude Code cannot drive a model without it, and
    that failure would otherwise only surface on the first turn.

    Dot-source this file to get the `cco` command, normally from your profile:

        . <path to this repo>\scripts\cco.ps1

    Running it instead of dot-sourcing defines the function inside the script's
    own scope and then discards it, which looks like nothing happened.

    See docs/how-to/powershell.md for the walkthrough, including how to pass
    Claude Code's own flags through. This is a port of the `cco` zsh function in
    https://github.com/paulocfjunior/shell-aliases.

.NOTES
    Exit codes: the child process sets $LASTEXITCODE as usual, so a caller can
    still test it after the session ends. A refusal from cco itself (bad flag,
    unknown model, no tools) writes to stderr and returns without running
    anything, leaving $LASTEXITCODE untouched.

    Windows support here has never been exercised on a real Windows machine, in
    this script or in ovp itself, so treat a failure there as a bug rather than a
    mistake of yours. See "Known limits" in README.md.
#>

function Invoke-ClaudeOllama {
    # No param() block on purpose. The flags here are `--double-dashed` to match
    # the zsh original and the tools being wrapped, and hand-parsing $args keeps
    # PowerShell's parameter binder from trying to resolve them as its own.

    function Write-CcoError {
        param([string]$Message)
        # Console stderr rather than Write-Error: this is a CLI refusal, not a
        # pipeline error, and an ErrorRecord would bury one line under a trace.
        [Console]::Error.WriteLine($Message)
    }

    function Show-CcoHelp {
        @(
            'Usage: cco <model> [--no-bridge|--bridge] [--vision-model M] [--proxy-port P] [-- claude args]'
            ''
            '  <model>            any model ollama knows, e.g. gemma4:e2b-mlx'
            '  --no-bridge        force straight through ollama, no image transcription'
            '  --bridge           force the ovp bridge even if the model has vision'
            "  --vision-model M   model that describes images (ovp's default is gemma3:4b)"
            '  --proxy-port P     pin the bridge to port P (default: a free one per call,'
            '                     so several sessions can run at the same time)'
            ''
            "By default the bridge is chosen from 'ollama show <model>':"
            '  reports vision      -> no bridge, ollama launch claude'
            '  no vision           -> bridge, via ovp'
            ''
            'Examples:'
            '  cco gemma4:e2b-mlx                      # no vision -> bridged'
            '  cco qwen3-vl:4b                         # has vision -> direct'
            '  cco gemma4:e2b-mlx -- --agent manager   # args after -- go to claude'
            '  cco gemma4:e2b-mlx --agent manager      # the -- is optional here'
            ''
            'Note: the bridge holds TWO models resident, the target and the vision'
            'one. On a tight machine prefer a vision-capable model with --no-bridge.'
            'Concurrent sessions share those resident models; each one only adds its'
            'own proxy process, so running several is cheap.'
        ) | ForEach-Object { Write-Host $_ }
    }

    function Get-CcoCapabilities {
        param([string]$Model)

        # Reads the manifest only; this does not load the model into memory.
        $output = & ollama show $Model 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) { return @() }

        $capabilities = @()
        $inSection = $false
        foreach ($line in $output) {
            if (-not $inSection) {
                if ($line -match 'Capabilities') { $inSection = $true }
                continue
            }
            # The section ends at the first blank line after it.
            if ([string]::IsNullOrWhiteSpace($line)) { break }
            $capabilities += ($line.Trim() -split '\s+')[0]
        }
        return $capabilities
    }

    $model = $null
    $bridge = 'auto'
    $visionModel = $null
    $proxyPort = $null
    $claudeArgs = @()

    $i = 0
    while ($i -lt $args.Count) {
        $arg = [string]$args[$i]

        if ($arg -eq '--no-bridge' -or $arg -eq '--no-vision') {
            $bridge = 'off'
            $i += 1
        }
        elseif ($arg -eq '--bridge') {
            $bridge = 'on'
            $i += 1
        }
        # The emptiness guards are not cosmetic. Their zsh counterparts stop a
        # `shift 2` that cannot consume two arguments, which would otherwise
        # leave the loop spinning on the same one forever.
        elseif ($arg -eq '--vision-model') {
            if ($i + 1 -ge $args.Count -or [string]::IsNullOrEmpty([string]$args[$i + 1])) {
                Write-CcoError 'cco: --vision-model needs a model name'
                return
            }
            $visionModel = [string]$args[$i + 1]
            $i += 2
        }
        elseif ($arg -eq '--proxy-port') {
            if ($i + 1 -ge $args.Count -or [string]::IsNullOrEmpty([string]$args[$i + 1])) {
                Write-CcoError 'cco: --proxy-port needs a port number'
                return
            }
            $proxyPort = [string]$args[$i + 1]
            $i += 2
        }
        elseif ($arg -eq '-h' -or $arg -eq '--help') {
            Show-CcoHelp
            return
        }
        elseif ($arg -eq '--') {
            # Guarded because $args[$n..($args.Count - 1)] counts backwards when
            # the -- is the last token, which would hand claude a reversed copy
            # of the whole command line.
            if ($i + 1 -lt $args.Count) {
                $claudeArgs += $args[($i + 1)..($args.Count - 1)]
            }
            break
        }
        else {
            if (-not $model) { $model = $arg } else { $claudeArgs += $arg }
            $i += 1
        }
    }

    if (-not $model) {
        Write-CcoError 'cco: no model given. Try: cco --help'
        return
    }

    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-CcoError 'cco: ollama is not on PATH. Install it from https://ollama.com'
        return
    }

    $caps = Get-CcoCapabilities -Model $model
    if ($caps.Count -eq 0) {
        Write-CcoError "cco: ollama does not know '$model'. Check the name with 'ollama ls', or pull it."
        return
    }

    $capList = $caps -join ','

    if ($caps -notcontains 'tools') {
        Write-CcoError "cco: '$model' reports [$capList] with no 'tools'."
        Write-CcoError '     Claude Code needs tool support and would fail on the first turn.'
        return
    }

    if ($bridge -eq 'auto') {
        $bridge = if ($caps -contains 'vision') { 'off' } else { 'on' }
    }

    # Arguments are splatted from an array rather than written inline so every
    # token, the bare -- included, reaches the child exactly as written instead
    # of going through PowerShell's own reading of it.
    if ($bridge -eq 'off') {
        Write-Host "cco: $model [$capList] -> direct, no vision bridge"
        $ollamaArgs = @('launch', 'claude', '--model', $model)
        if ($claudeArgs.Count -gt 0) {
            $ollamaArgs += '--'
            $ollamaArgs += $claudeArgs
        }
        & ollama @ollamaArgs
        return
    }

    if (-not (Get-Command ovp -ErrorAction SilentlyContinue)) {
        Write-CcoError "cco: '$model' has no vision and ovp is not installed, so pasted images would be rejected."
        Write-CcoError '     Install it:  uv tool install ollama-vision-proxy'
        Write-CcoError "     Or skip it:  cco $model --no-bridge"
        return
    }

    # Port 0 lets the OS hand out a free port, and ovp reads the bound port back
    # before it points claude at it, so several sessions coexist. ovp defaults to
    # this itself; passing it keeps the behaviour with an older ovp installed,
    # which would otherwise ask for its fixed 11435 and make the second
    # concurrent session die with "Address already in use".
    $port = if ($proxyPort) { $proxyPort } else { '0' }
    $ovpArgs = @('launch', '--target-model', $model, '--proxy-port', $port)
    if ($visionModel) { $ovpArgs += @('--vision-model', $visionModel) }
    if ($claudeArgs.Count -gt 0) {
        $ovpArgs += '--'
        $ovpArgs += $claudeArgs
    }

    Write-Host "cco: $model [$capList] -> via ovp, images transcribed locally"
    & ovp @ovpArgs
}

Set-Alias -Name cco -Value Invoke-ClaudeOllama
