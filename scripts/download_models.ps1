# PowerShell Model Download Script for Spam Detection Models
# Downloads trained models from Google Drive to the models/ directory
#
# Usage:
#   .\scripts\download_models.ps1                    # Download all models
#   .\scripts\download_models.ps1 -Models svm       # Download specific model  
#   .\scripts\download_models.ps1 -List             # List available models

param(
    [string[]]$Models = @(),
    [switch]$List = $false,
    [string]$ModelsDir = "models"
)

# Model configuration
$ModelsConfig = @{
    'svm' = @{
        'filename' = 'svm_full.pkl'
        'url' = 'https://drive.google.com/uc?id=1Vxjz4QV3FESvm7gMeKNqklA5uARPntLv'
        'description' = 'Support Vector Machine classifier'
        'size' = '~15MB'
    }
    'enhanced_transformer' = @{
        'filename' = 'enhanced_transformer_99recall.pt'
        'url' = 'https://drive.google.com/uc?id=1kGD6Tg5JLIko0XhYPk2-WtAswj1S2Pgs'
        'description' = 'Enhanced Transformer model (99% spam recall)'
        'size' = '~172MB'
    }
    'catboost' = @{
        'filename' = 'catboost_tuned.pkl'
        'url' = 'https://drive.google.com/uc?id=1ofS_IU9QiypgkvFqNGLjUvSdUfEi9hjO'
        'description' = 'Tuned CatBoost classifier'
        'size' = '~10MB'
    }
}

function Write-ColorText {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Test-PythonGdown {
    try {
        python -c "import gdown" 2>$null
        return $true
    }
    catch {
        return $false
    }
}

function Download-Model {
    param(
        [string]$ModelName,
        [hashtable]$Config,
        [string]$OutputDir
    )
    
    $filename = $Config.filename
    $url = $Config.url  
    $description = $Config.description
    $size = $Config.size
    
    $outputPath = Join-Path $OutputDir $filename
    
    Write-ColorText "`n📥 Downloading $($ModelName.ToUpper()) model:" "Cyan"
    Write-ColorText "   📄 $description" "Gray"
    Write-ColorText "   📦 Size: $size" "Gray"  
    Write-ColorText "   🎯 Target: $outputPath" "Gray"
    
    if (Test-Path $outputPath) {
        Write-ColorText "   ✅ File already exists: $outputPath" "Green"
        return $true
    }
    
    try {
        Write-ColorText "   🔄 Downloading from Google Drive..." "Yellow"
        
        # Try using Python gdown first
        if (Test-PythonGdown) {
            python -c "import gdown; gdown.download('$url', '$outputPath', quiet=False)"
        } else {
            # Fallback to PowerShell web request  
            Write-ColorText "   ⚠️  Python gdown not available, using PowerShell..." "Yellow"
            Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing
        }
        
        if (Test-Path $outputPath) {
            $fileSize = (Get-Item $outputPath).Length / 1MB
            Write-ColorText "   ✅ Download successful! ($([math]::Round($fileSize, 1)) MB)" "Green"
            return $true
        } else {
            Write-ColorText "   ❌ Download failed - file not created" "Red"
            return $false
        }
    }
    catch {
        Write-ColorText "   ❌ Download failed: $($_.Exception.Message)" "Red"
        return $false
    }
}

# Main logic
Write-ColorText "🤖 Spam Detection Models Downloader" "Cyan"

if ($List) {
    Write-ColorText "`n📋 Available models:" "Cyan"
    foreach ($name in $ModelsConfig.Keys) {
        $info = $ModelsConfig[$name]
        Write-ColorText "   • $($name.PadRight(20)) - $($info.description) ($($info.size))" "Gray"
    }
    exit 0
}

# Setup models directory
if (!(Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
}

$absoluteDir = (Resolve-Path $ModelsDir).Path
Write-ColorText "🎯 Models directory: $absoluteDir" "Gray"

# Determine which models to download
if ($Models.Count -gt 0) {
    $requestedModels = $Models
    $invalid = $requestedModels | Where-Object { $_ -notin $ModelsConfig.Keys }
    if ($invalid.Count -gt 0) {
        Write-ColorText "❌ Invalid model names: $($invalid -join ', ')" "Red"
        Write-ColorText "✅ Available models: $($ModelsConfig.Keys -join ', ')" "Green"
        exit 1
    }
} else {
    $requestedModels = $ModelsConfig.Keys
    Write-ColorText "📦 Downloading all models: $($requestedModels -join ', ')" "Gray"
}

# Check dependencies
if (!(Test-PythonGdown)) {
    Write-ColorText "`n⚠️  Recommended: Install Python gdown for better download reliability" "Yellow"
    Write-ColorText "   pip install gdown" "Gray"
    Write-ColorText "   Falling back to PowerShell web requests..." "Gray"
}

# Download models
$successful = @()
$failed = @()

foreach ($modelName in $requestedModels) {
    if (Download-Model -ModelName $modelName -Config $ModelsConfig[$modelName] -OutputDir $ModelsDir) {
        $successful += $modelName
    } else {
        $failed += $modelName
    }
}

# Summary
Write-ColorText "`n$('='*50)" "Gray"
Write-ColorText "📊 Download Summary:" "Cyan"

if ($successful.Count -gt 0) {
    Write-ColorText "   ✅ Successful ($($successful.Count)): $($successful -join ', ')" "Green"
}

if ($failed.Count -gt 0) {
    Write-ColorText "   ❌ Failed ($($failed.Count)): $($failed -join ', ')" "Red"
}

Write-ColorText "`n🎯 Models location: $absoluteDir" "Gray"

if ($successful.Count -gt 0) {
    Write-ColorText "`n🚀 Ready to use! Try running:" "Green"
    Write-ColorText "   python predictors/predict_main.py `"Test spam message`"" "Gray"
}

if ($failed.Count -gt 0) {
    exit 1
} else {
    exit 0
}