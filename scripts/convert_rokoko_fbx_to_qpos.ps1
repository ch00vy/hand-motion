param(
    [Parameter(Mandatory = $true)][string]$InputFbx,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateSet("right", "left")][string]$Hand = "right",
    [ValidateSet("open_to_fist", "fist_to_open")][string]$MotionType = "open_to_fist",
    [string]$SubjectId = "subject_unknown",
    [ValidateSet("normal", "abnormal", "testing")][string]$TrialLabel = "normal"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$blender = Get-ChildItem 'C:\Program Files\Blender Foundation' -Recurse -Filter blender.exe -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $blender) { throw "Blender tidak ditemukan. Pasang Blender terlebih dahulu." }

& $blender --background --python (Join-Path $PSScriptRoot 'convert_rokoko_fbx_to_qpos_blender.py') -- `
    --input $InputFbx --output-dir $OutputDir --hand $Hand --motion-type $MotionType `
    --subject-id $SubjectId --trial-label $TrialLabel
if ($LASTEXITCODE -ne 0) { throw "Konversi FBX gagal (exit code $LASTEXITCODE)." }
