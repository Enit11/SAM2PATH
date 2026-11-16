$ErrorActionPreference = "Stop"
$root = "C:\Users\ASUS\SAM2PATH"
$pythonExe = "C:\Users\ASUS\.conda\envs\sam2path_gpu\python.exe"
$checkpoint = Join-Path $root "outputs\glomeruli\run\glomeruli_uni2_spherical_hrrefine_warmstart\Logs\20261013-150230\sam2path\version_0\checkpoints\last.ckpt"
$imageDir = Join-Path $root "dataset_patient_split_seed42\test\image"
$maskDir = Join-Path $root "dataset_patient_split_seed42\test\mask"
$outputDir = Join-Path $root "outputs\bridge_diagnosis\spherical_hrrefine_epoch046_seed42_20"

& $pythonExe scripts\diagnose_same_class_bridges.py `
  --checkpoint $checkpoint `
  --image-dir $imageDir `
  --mask-dir $maskDir `
  --output-dir $outputDir `
  --config configs.Glomeruli_uni2_spherical_hrrefine `
  --project-root $root `
  --num-samples 20 `
  --seed 42 `
  --bridge-radius 24
