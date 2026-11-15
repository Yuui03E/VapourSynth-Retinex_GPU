param(
    [switch]$CpuOnly,
    [switch]$Testing,
    [string]$Architectures = '75-real;86-real;89-real;120-real;75-virtual',
    [string]$BuildDirectory = '',
    [string]$CudaCompiler = '',
    [switch]$References
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $BuildDirectory) { $BuildDirectory = if ($CpuOnly) { 'build/cpu-only' } elseif ($Testing) { 'build/testing' } else { 'build/all-in-one' } }
# Import the installed MSVC environment into this process. No visible helper window.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsRoot) { throw 'Install Visual Studio Build Tools with Desktop development with C++.' }
    $devCmd = Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
    $envLines = & $env:ComSpec /d /s /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
}
$configureArgs = @('-S', '.', '-B', $BuildDirectory, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
if (-not $CpuOnly -and -not $CudaCompiler) {
    $localCompiler = Join-Path $projectRoot '.tools\cuda\nvidia\cu13\bin\nvcc.exe'
    if ((-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) -and (Test-Path -LiteralPath $localCompiler)) {
        $CudaCompiler = $localCompiler
    }
}
if (-not $CpuOnly -and $CudaCompiler) { $configureArgs += "-DCMAKE_CUDA_COMPILER=$CudaCompiler" }
$configureArgs += "-DRETINEX_CUDA=$(if ($CpuOnly) { 'OFF' } else { 'ON' })"
$configureArgs += "-DRETINEX_TESTING=$(if ($Testing) { 'ON' } else { 'OFF' })"
if (-not $CpuOnly) { $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$Architectures" }
$configureArgs += "-DBUILD_CPU_REFERENCES=$(if ($References) { 'ON' } else { 'OFF' })"
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build $BuildDirectory --parallel
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Output (Join-Path $projectRoot "$BuildDirectory\retinex_gpu.dll")
