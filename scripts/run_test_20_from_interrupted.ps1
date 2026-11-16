$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\ASUS\.conda\envs\sam2path_gpu\python.exe"
$projectRoot = "C:\Users\ASUS\SAM2PATH"
$checkpoint = Join-Path $projectRoot "outputs\glomeruli\run\glomeruli_uni2_spherical_hrrefine_warmstart\Logs\20261013-150230\sam2path\version_0\checkpoints\last.ckpt"
$imageDir = Join-Path $projectRoot "dataset_patient_split_seed42\test\image"
$maskDir = Join-Path $projectRoot "dataset_patient_split_seed42\test\mask"
$outputDir = Join-Path $projectRoot "vis_compare\spherical_hrrefine_interrupted_last_20"

if (!(Test-Path -LiteralPath $checkpoint)) { throw "Checkpoint not found: $checkpoint" }
if (!(Test-Path -LiteralPath $imageDir)) { throw "Test image directory not found: $imageDir" }
if (!(Test-Path -LiteralPath $maskDir)) { throw "Test mask directory not found: $maskDir" }

Set-Location $projectRoot
& $pythonExe -c "from sam2path_inference import run_inference; ok=run_inference(ckpt_path=r'$checkpoint', image_dir=r'$imageDir', mask_dir=r'$maskDir', output_dir=r'$outputDir', num_samples=20, seed=42, project_root=r'$projectRoot', config_module='configs.Glomeruli_uni2_spherical_hrrefine'); raise SystemExit(0 if ok else 1)"
