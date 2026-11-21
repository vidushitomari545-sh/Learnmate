$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$BackendDirectory = Join-Path $ProjectRoot "backend"
$FrontendDirectory = Join-Path $ProjectRoot "frontend"
$PythonExecutable = Join-Path $BackendDirectory ".venv\Scripts\python.exe"
$BackendPort = 8765
$FrontendPort = 5173
$OllamaBaseUrl = "http://127.0.0.1:11434"
$OllamaModel = "llama3.2:3b"
$BackendUrl = "http://127.0.0.1:$BackendPort"
$FrontendUrl = "http://127.0.0.1:$FrontendPort"

function Test-TcpPort {
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    $pendingConnection = $null
    try {
        $pendingConnection = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $pendingConnection.AsyncWaitHandle.WaitOne(500)) {
            return $false
        }
        $client.EndConnect($pendingConnection)
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $pendingConnection) {
            $pendingConnection.AsyncWaitHandle.Close()
        }
        $client.Dispose()
    }
}

function Test-OllamaApi {
    try {
        $null = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -Method Get -TimeoutSec 2
        return $true
    }
    catch {
        return $false
    }
}

function Test-BackendHealth {
    try {
        $health = Invoke-RestMethod -Uri "$BackendUrl/api/health" -Method Get -TimeoutSec 2
        $api = Invoke-RestMethod -Uri "$BackendUrl/openapi.json" -Method Get -TimeoutSec 2
        $paths = @($api.paths.PSObject.Properties | ForEach-Object { $_.Name })
        return $health.status -eq "ok" -and $paths -contains "/api/documents/{document_id}/summary"
    }
    catch {
        return $false
    }
}

function Test-LearnMateFrontend {
    try {
        $response = Invoke-WebRequest -Uri $FrontendUrl -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -ne 200 -or $response.Content -notmatch "<title>\s*LearnMate") {
            return $false
        }

        try {
            $null = Invoke-RestMethod `
                -Uri "$FrontendUrl/api/documents/startup-probe/summary/startup-probe" `
                -Method Get `
                -TimeoutSec 2
            return $false
        }
        catch {
            $statusCode = 0
            if ($null -ne $_.Exception.Response) {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
            return $statusCode -eq 404 -and $_.ErrorDetails.Message -match "Document summary not found"
        }
    }
    catch {
        return $false
    }
}

function Wait-ForService {
    param(
        [scriptblock]$Probe,
        [int]$TimeoutSeconds,
        [string]$ServiceName
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Probe) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$ServiceName did not become ready within $TimeoutSeconds seconds. Check its service window for errors, then run this script again."
}

$startupMutex = New-Object System.Threading.Mutex($false, "Local\LearnMateDevStartup")
$ownsStartupMutex = $false
try {
    try {
        $ownsStartupMutex = $startupMutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $ownsStartupMutex = $true
    }
    if (-not $ownsStartupMutex) {
        throw "Another LearnMate startup is already in progress. Wait for it to finish, then check the service windows."
    }

    if (-not (Test-OllamaApi)) {
        $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -eq $ollamaCommand) {
            throw "Ollama was not found on PATH. Install Ollama for Windows, then run this script again."
        }

        Write-Host "Ollama is not responding. Starting 'ollama serve' in a separate window..."
        $null = Start-Process -FilePath $ollamaCommand.Source -ArgumentList "serve" -PassThru
        try {
            Wait-ForService -Probe { Test-OllamaApi } -TimeoutSeconds 60 -ServiceName "Ollama"
        }
        catch {
            throw "$($_.Exception.Message) Ollama may already be running but not ready; check its window and http://127.0.0.1:11434."
        }
    }
    else {
        Write-Host "Ollama is responding at $OllamaBaseUrl."
    }

    try {
        $ollamaTags = Invoke-RestMethod -Uri "$OllamaBaseUrl/api/tags" -Method Get -TimeoutSec 5
    }
    catch {
        throw "Ollama responded but its model list could not be verified. Check the Ollama service and try again."
    }
    $installedModel = @(
        $ollamaTags.models | Where-Object {
            $_.name -eq $OllamaModel -or $_.model -eq $OllamaModel
        }
    )
    if ($installedModel.Count -eq 0) {
        throw "Required model '$OllamaModel' is not installed. Install it manually with: ollama pull $OllamaModel. No model was downloaded."
    }
    Write-Host "Verified local model: $OllamaModel."

    if (Test-BackendHealth) {
        Write-Host "LearnMate backend is already responding at $BackendUrl; leaving it running."
    }
    else {
        if (Test-TcpPort -Port $BackendPort) {
            throw "Port $BackendPort is in use, but it is not serving the current LearnMate API. Stop the conflicting process or configure the project port before retrying."
        }
        if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
            throw "The existing backend interpreter was not found at '$PythonExecutable'. Create backend\.venv and install backend\requirements-dev.txt first."
        }

        Write-Host "Starting the FastAPI backend on port $BackendPort in a separate window..."
        $providerVariables = @{
            LEARNMATE_LLM_PROVIDER = "ollama"
            LEARNMATE_OLLAMA_BASE_URL = $OllamaBaseUrl
            LEARNMATE_OLLAMA_MODEL = $OllamaModel
        }
        $previousProviderVariables = @{}
        foreach ($name in $providerVariables.Keys) {
            $previousProviderVariables[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
            [Environment]::SetEnvironmentVariable($name, $providerVariables[$name], "Process")
        }
        try {
            $null = Start-Process `
                -FilePath $PythonExecutable `
                -ArgumentList @(
                    "-m", "uvicorn", "app.main:app", "--reload",
                    "--host", "127.0.0.1", "--port", "$BackendPort"
                ) `
                -WorkingDirectory $BackendDirectory `
                -PassThru
        }
        finally {
            foreach ($name in $providerVariables.Keys) {
                [Environment]::SetEnvironmentVariable(
                    $name,
                    $previousProviderVariables[$name],
                    "Process"
                )
            }
        }
        Wait-ForService -Probe { Test-BackendHealth } -TimeoutSeconds 60 -ServiceName "LearnMate backend"
    }

    if (Test-LearnMateFrontend) {
        Write-Host "LearnMate frontend is already responding at $FrontendUrl; leaving it running."
    }
    else {
        if (Test-TcpPort -Port $FrontendPort) {
            throw "Port $FrontendPort is in use, but it is not serving the current LearnMate frontend/API proxy. Restart its Vite window or stop the conflicting process before retrying."
        }
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if ($null -eq $npmCommand) {
            throw "npm was not found on PATH. Install Node.js and npm, then run this script again."
        }

        Write-Host "Starting the Vite frontend on port $FrontendPort in a separate window..."
        $null = Start-Process `
            -FilePath $env:ComSpec `
            -ArgumentList @("/k", "npm run dev -- --host 127.0.0.1 --strictPort") `
            -WorkingDirectory $FrontendDirectory `
            -PassThru
        Wait-ForService -Probe { Test-LearnMateFrontend } -TimeoutSeconds 45 -ServiceName "LearnMate frontend"
    }

    Write-Host ""
    Write-Host "LearnMate is ready:"
    Write-Host "  Frontend: $FrontendUrl"
    Write-Host "  Backend:  $BackendUrl"
    Write-Host "  API docs: $BackendUrl/docs"
    Write-Host "The backend uses Ollama's HTTP API; it does not run interactive 'ollama run'."
    Write-Host "To stop a service, press Ctrl+C in that service's window. Stop the Ollama window only if this script started it."
}
finally {
    if ($ownsStartupMutex) {
        $startupMutex.ReleaseMutex()
    }
    $startupMutex.Dispose()
}
