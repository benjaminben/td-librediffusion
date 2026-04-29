# Reports which librediffusion / TRT / CUDA DLLs TouchDesigner has actually loaded.
# Run while TD is open with our Custom Operator TOP active in a project.
#
#   powershell -ExecutionPolicy Bypass -File check-td-dlls.ps1

$proc = Get-Process TouchDesigner -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Host "TouchDesigner is not running" -ForegroundColor Red
    exit 1
}

Write-Host "=== TouchDesigner modules matching librediffusion / TRT / CUDA ===" -ForegroundColor Cyan
$proc.Modules `
    | Where-Object {
        $_.ModuleName -like "*librediffusion*" -or
        $_.ModuleName -like "*nvinfer*" -or
        $_.ModuleName -like "*cudart*" -or
        $_.ModuleName -like "*curand*" -or
        $_.ModuleName -like "*cublas*" -or
        $_.ModuleName -like "*npp*" -or
        $_.ModuleName -like "*td_librediffusion*"
    } `
    | Select-Object ModuleName, FileName `
    | Format-Table -AutoSize

Write-Host "=== TouchDesigner PID + memory ===" -ForegroundColor Cyan
"PID:                   {0}" -f $proc.Id
"Working Set (RAM):     {0:N0} MB" -f ($proc.WorkingSet64 / 1MB)
"Private Memory:        {0:N0} MB" -f ($proc.PrivateMemorySize64 / 1MB)
