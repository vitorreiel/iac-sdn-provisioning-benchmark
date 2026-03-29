#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=================================================================="
echo "Shapiro-Wilk Normality Tests — IaC Tools Benchmark"
echo "=================================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found"
    exit 1
fi

python3 << 'PYEOF'
try:
    import pandas, matplotlib, seaborn, numpy, scipy
    print("✓ All dependencies are installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install pandas matplotlib seaborn numpy scipy")
    exit(1)
PYEOF

echo ""
python3 "$SCRIPT_DIR/normality_test.py"

OUTPUT_DIR="$SCRIPT_DIR/outputs"
if [ -d "$OUTPUT_DIR" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
    CSV=$(ls -1 "$OUTPUT_DIR"/*.csv  2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ] || [ "$CSV" -gt 0 ]; then
        echo ""
        echo "=================================================================="
        echo "✓ Generated $COUNT figures + $CSV CSV files"
        echo "=================================================================="
        echo ""
        echo "Output location: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -1 "$OUTPUT_DIR"/ 2>/dev/null | xargs -n1 echo " -"
        echo ""
    fi
fi
