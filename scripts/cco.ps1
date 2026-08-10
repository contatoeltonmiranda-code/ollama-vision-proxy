#Requires -Version 5.1
<#
.SYNOPSIS
    PowerShell front end for ollama-vision-proxy.

.DESCRIPTION
    Dot-source this file from your PowerShell profile to get `cco`: Claude Code
    running against an Ollama model, with images transcribed by a local vision
    model on the way through.

        . "$HOME\ollama-vision-proxy\scripts\cco.ps1"

    Nothing here writes to the environment of your session. ANTHROPIC_BASE_URL
    is set by ovp inside the child process only, so a plain `claude` in the same
    window, or in any other window, still reaches Anthropic exactly as before.
    That is why this wraps `ovp launch` instead of assigning a few `$env:`
    variables: an assignment would leak to every later command in the session,
    and `claude` would silently start talking to Ollama.

    Every invocation gets its own proxy on an OS-assigned port, so as many
    sessions as you like can run side by side. Pin -ProxyPort only if something
    outside the session has to reach the proxy, and never in two windows at once.

.EXAMPLE
    cco
    Starts a session against the default target model, with vision support.

.EXAMPLE
    cco --resume
    Anything you pass is appended to the claude command line, after the
    defaults, so any claude flag works here.

.EXAMPLE
    Invoke-ClaudeOllama -VisionModel qwen3-vl:4b -LogFile $env:TEMP\ovp.log
    The long form, for when a default needs changing.

.EXAMPLE
    Invoke-ClaudeOllama -SkipPermissions
    Passes --dangerously-skip-permissions. Off by default: routing at a local
    model is not on its own a reason to stop approving tool calls, and this
    file is dot-sourced from a profile where that would be easy to forget.
#>

# The model Claude Code actually talks to. It must report the tools capability.
$script:OvpTargetModel = 'glm-5.2:cloud'

# The local model that describes images. Must report the vision capability.
$script:OvpVisionModel = 'gemma3:4b'

# Claude arguments added to every session started through this wrapper. Empty by
# design: what belongs here is personal (an --agent, a --channels plugin, a
# --model alias), and a default that names something only present on one machine
# fails for everyone else. Add your own, they land before anything passed at the
# call site.
#
# --model is deliberately absent rather than empty: ovp points all three of
# claude's model aliases at the target model, so the default alias already
# resolves there without naming it.
$script:OvpClaudeArgs = @()

$script:OvpUpstream = 'http://127.0.0.1:11434'

# Variables that redirect Claude Code away from Anthropic. `ollama launch claude`
# sets these, and so does ovp, and a value inherited by the wrong process is
# invisible: claude simply talks to a port that is not listening any more.
$script:OvpRedirectVars = @(
    'ANTHROPIC_BASE_URL'
    'ANTHROPIC_AUTH_TOKEN'
    'ANTHROPIC_DEFAULT_OPUS_MODEL'
    'ANTHROPIC_DEFAULT_SONNET_MODEL'
    'ANTHROPIC_DEFAULT_HAIKU_MODEL'
    'CLAUDE_CODE_SUBAGENT_MODEL'
)


function Clear-ClaudeRedirect {
    # Strip any inherited redirect so a plain `claude` reaches Anthropic.
    #
    # This exists because the variables outlive the thing that set them. A
    # session started by `ollama launch claude` carries
    # ANTHROPIC_BASE_URL=http://127.0.0.1:11434 in its own environment, and every
    # terminal, script and claude opened from inside it inherits the value. Once
    # that session is gone the port answers nothing, and the symptom is every
    # later claude failing to connect, in windows that look unrelated. The same
    # applies to a value written to the persistent User environment by any tool.
    #
    # A blank ANTHROPIC_API_KEY is cleared too. ovp sets it blank on purpose in
    # its child, to neutralise an inherited real key; inherited one level
    # further it only makes Claude Code announce that connectors are disabled.
    # A key with an actual value is left alone, since that one may be deliberate.
    [CmdletBinding()]
    param([switch]$Quiet)

    $cleared = @()

    foreach ($name in $script:OvpRedirectVars) {
        if (Test-Path "Env:$name") {
            Remove-Item "Env:$name"
            $cleared += $name
        }
        if ([Environment]::GetEnvironmentVariable($name, 'User')) {
            [Environment]::SetEnvironmentVariable($name, $null, 'User')
            $cleared += "$name (persistent)"
        }
    }

    if ((Test-Path 'Env:ANTHROPIC_API_KEY') -and -not $env:ANTHROPIC_API_KEY) {
        Remove-Item 'Env:ANTHROPIC_API_KEY'
        $cleared += 'ANTHROPIC_API_KEY (blank)'
    }

    if ($cleared.Count -and -not $Quiet) {
        Write-Host "ovp: cleared an inherited redirect: $($cleared -join ', ')" -ForegroundColor DarkGray
    }
    return $cleared
}

function Find-OvpCommand {
    # Locate the ovp executable. PATH first, then the two places an install from
    # a clone tends to land, so a working setup does not depend on whether
    # `uv tool update-shell` was ever run.
    $onPath = Get-Command ovp -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $candidates = @(
        (Join-Path $PSScriptRoot '..\.venv\Scripts\ovp.exe'),
        (Join-Path $HOME '.local\bin\ovp.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}


function Test-OvpOllama {
    # True when the Ollama server answers.
    try {
        $null = Invoke-RestMethod -Uri "$script:OvpUpstream/api/version" -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}


function Get-OvpModelCapability {
    # Capabilities reported by /api/show, or an empty array when Ollama does not
    # know the model. An empty array is the "not pulled" signal; it is not the
    # same as a model that is present but lacks the capability you wanted.
    param([Parameter(Mandatory = $true)][string]$Model)

    try {
        $body = @{ model = $Model } | ConvertTo-Json -Compress
        $shown = Invoke-RestMethod -Uri "$script:OvpUpstream/api/show" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 15
    } catch {
        return @()
    }
    if (-not $shown.PSObject.Properties.Match('capabilities').Count) { return @() }
    return @($shown.capabilities)
}


function Confirm-OvpModel {
    # Make sure a model is present and has the capability its role needs,
    # pulling it if it is missing. Returns $true when the model is usable.
    #
    # Both checks matter and they fail at different times. A missing model
    # surfaces as a 404 on the first turn; a model that is present but lacks the
    # capability surfaces later still, as a 400 that reads like a bug in the
    # proxy. Catching both here means the session either starts correct or does
    # not start.
    param(
        [Parameter(Mandatory = $true)][string]$Model,
        [Parameter(Mandatory = $true)][string]$Capability,
        [switch]$Yes
    )

    $capabilities = Get-OvpModelCapability -Model $Model
    if ($capabilities -contains $Capability) { return $true }

    if ($capabilities.Count -gt 0) {
        Write-Host "ovp: '$Model' is available but does not support '$Capability'." -ForegroundColor Red
        Write-Host "ovp: it reports: $($capabilities -join ', ')" -ForegroundColor Red
        return $false
    }

    Write-Host "ovp: model '$Model' is not available locally." -ForegroundColor Yellow
    if (-not $Yes) {
        # Only ask when there is someone to answer. ovp's own prompt checks
        # stdin.isatty() first for the same reason: an unattended run (a
        # scheduled task, CI, an agent) would otherwise block here forever
        # rather than failing with a message someone can read later.
        if (-not [Environment]::UserInteractive) {
            Write-Host "ovp: not interactive, so not prompting. Run 'ollama pull $Model', or pass -Yes." -ForegroundColor Red
            return $false
        }
        $answer = Read-Host "ovp: run 'ollama pull $Model' now? [y/N]"
        if ($answer -notmatch '^(y|yes)$') {
            Write-Host "ovp: cannot continue without '$Model'." -ForegroundColor Red
            return $false
        }
    }

    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ovp: 'ollama pull $Model' failed." -ForegroundColor Red
        return $false
    }

    $capabilities = Get-OvpModelCapability -Model $Model
    return ($capabilities -contains $Capability)
}


function Invoke-ClaudeOllama {
    # Start Claude Code against an Ollama model, with image support.
    # -ClaudeArgs is appended to the claude command line, after the defaults.
    [CmdletBinding()]
    param(
        [string]$TargetModel = $script:OvpTargetModel,
        [string]$VisionModel = $script:OvpVisionModel,
        [int]$ProxyPort = 0,
        [string]$LogFile,
        [switch]$Yes,
        [switch]$Trace,
        [switch]$SkipPermissions,
        [string[]]$ClaudeArgs = @()
    )

    # A stale redirect in this session would not break ovp, but it would break
    # the next plain claude run in the same window. Clear it while we are here.
    $null = Clear-ClaudeRedirect -Quiet

    $ovp = Find-OvpCommand
    if (-not $ovp) {
        Write-Host "ovp: the ovp command was not found." -ForegroundColor Red
        Write-Host "ovp: install it from a clone of the repo, for example:" -ForegroundColor Red
        Write-Host "     py -m venv .venv; .venv\Scripts\python -m pip install ." -ForegroundColor Red
        return
    }

    if (-not (Test-OvpOllama)) {
        Write-Host "ovp: cannot reach the Ollama server at $script:OvpUpstream." -ForegroundColor Red
        Write-Host "ovp: start it with 'ollama serve' and try again." -ForegroundColor Red
        return
    }

    # The target model needs tools to drive Claude Code at all; the vision model
    # needs vision to describe anything. ovp launch only preflights the second
    # one, so both are checked here, before anything is spawned.
    if (-not (Confirm-OvpModel -Model $TargetModel -Capability 'tools' -Yes:$Yes)) { return }
    if (-not (Confirm-OvpModel -Model $VisionModel -Capability 'vision' -Yes:$Yes)) { return }

    # Built as one array and splatted. PowerShell leaves a bare -- alone for
    # native commands, but splatting removes the question entirely: every element
    # reaches ovp as its own argv entry, whatever the parser mode.
    $argv = @(
        'launch'
        '--target-model', $TargetModel
        '--vision-model', $VisionModel
    )
    if ($ProxyPort -gt 0) { $argv += @('--proxy-port', "$ProxyPort") }
    if ($LogFile)         { $argv += @('--log-file', $LogFile) }
    if ($Trace)           { $argv += '--verbose' }

    # What claude receives, in order: the defaults from the top of this file,
    # then -SkipPermissions if it was asked for, then whatever the call site
    # passed. The bare -- goes in only when there is something to put after it.
    $claudeSide = @($script:OvpClaudeArgs)
    if ($SkipPermissions) { $claudeSide += '--dangerously-skip-permissions' }
    if ($ClaudeArgs.Count -gt 0) { $claudeSide += $ClaudeArgs }
    if ($claudeSide.Count -gt 0) {
        $argv += '--'
        $argv += $claudeSide
    }

    & $ovp @argv
}


function cco {
    # The everyday entry point. Arguments are forwarded verbatim, so
    # `cco --resume` and `cco -p "..."` both behave the way they would on the
    # claude command line, and a claude flag this wrapper knows nothing about
    # still reaches it. Passing $args as an explicit array rather than splatting
    # keeps PowerShell from binding a claude flag such as -p to a parameter of
    # Invoke-ClaudeOllama.
    Invoke-ClaudeOllama -ClaudeArgs $args
}
