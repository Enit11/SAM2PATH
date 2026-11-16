param(
    [string]$Python = "C:\Users\ASUS\.conda\envs\sam2path_gpu\python.exe",
    [string]$InitCheckpoint = "C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_uni2_resume2\Logs\20260916-145447\sam2path\version_0\checkpoints\epoch=64-step=10010.ckpt"
)

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\ASUS\SAM2PATH"

& $Python main.py --config configs.Glomeruli_s0_distance3ch_short --devices 0 --project sampath --name sam2uni2_s0_distance3ch_short --init_ckpt_path $InitCheckpoint
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python main.py --config configs.Glomeruli_s2_spherical_sep_short --devices 0 --project sampath --name sam2uni2_s2_spherical_sep_short --init_ckpt_path $InitCheckpoint
exit $LASTEXITCODE
