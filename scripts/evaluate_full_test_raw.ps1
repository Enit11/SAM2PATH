$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\ASUS\.conda\envs\sam2path_gpu\python.exe"
$root = "C:\Users\ASUS\SAM2PATH"
$checkpoint = Join-Path $root "outputs\glomeruli\run\glomeruli_uni2_spherical_hrrefine_warmstart\Logs\20261013-150230\sam2path\version_0\checkpoints\last.ckpt"
$imageDir = Join-Path $root "dataset_patient_split_seed42\test\image"
$maskDir = Join-Path $root "dataset_patient_split_seed42\test\mask"
$predDir = Join-Path $root "outputs\test_eval\spherical_hrrefine_epoch046_raw\predictions"
$metricDir = Join-Path $root "outputs\test_eval\spherical_hrrefine_epoch046_raw\metrics"

Set-Location $root
& $pythonExe scripts\evaluate_full_test_raw.py --checkpoint $checkpoint --image-dir $imageDir --mask-dir $maskDir --prediction-dir $predDir --config configs.Glomeruli_uni2_spherical_hrrefine --project-root $root
& $pythonExe evaluate_segmentation.py --pred-dir $predDir --gt-dir $maskDir --output-dir $metricDir --resolution 1024
