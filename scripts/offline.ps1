# Use only after prepare_data / an online first training has downloaded assets.
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
Write-Host "Hugging Face offline mode enabled in this terminal."
