# ----------------------------
# PowerShell script to run JMH benchmarks with checks
# ----------------------------

function Check-ExitCode {
    param($stepDescription)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $stepDescription completed successfully." -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $stepDescription failed with exit code $LASTEXITCODE." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 1️⃣ Run generate_matrices.py
Write-Host "`n============================================================"
Write-Host "Step 1: Generating matrices..."
Write-Host "============================================================`n"
python .\generate_matrices.py
Check-ExitCode "Matrix generation"

# 2 Compile Benchmark.java
Write-Host "`n============================================================"
Write-Host "Step 2: Compiling Benchmark.java..."
Write-Host "============================================================`n"
#javac .\Benchmark.java
javac --add-modules jdk.incubator.vector Benchmark.java
Check-ExitCode "Benchmark compilation"

# 3 Execute Benchmark.java
Write-Host "`n============================================================"
Write-Host "Step 3: Running Benchmark..."
Write-Host "============================================================`n"
#java -Xmx4G -XX:+TieredCompilation -XX:ActiveProcessorCount=16 Benchmark
java -Xmx4G -XX:+TieredCompilation -XX:ActiveProcessorCount=16 --add-modules jdk.incubator.vector Benchmark
Check-ExitCode "Benchmark execution"

Write-Host "`n[OK] All steps completed successfully." -ForegroundColor Green


