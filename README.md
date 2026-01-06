# Dot Array Stimulus Toolbox (DAST)

An open-source Python toolbox for generating and analyzing dot array stimuli used in numerical cognition research.
Preprint here: https://osf.io/preprints/psyarxiv/uhsv6_v1

## 🌐 Use Online (No Installation Required)

- **[Stimulus Generator](https://dot-array-stimulus-toolbox-generator.streamlit.app)** — Create dot arrays with controlled visual parameters
- **[Stimulus Analyzer](https://dot-array-stimulus-toolbox-analyzer.streamlit.app)** — Extract visual parameters from existing dot array images

## Features

### Generator
- Create dot array stimuli with precise control over numerosity, element size, cumulative area, and spatial distribution
- Export ground truth metrics for all generated stimuli
- Supports uniform or variable dot sizes
- Batch generation with CSV export

### Analyzer
- Extract visual parameters from any dot array image
- Measures: numerosity, cumulative area, average element size, convex hull area, density, nearest neighbor distance, and more
- Visual preview of detected elements
- Batch processing with CSV export

## Local Installation

If you prefer to run the tools locally:

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/dot-array-stimulus-toolbox.git
cd dot-array-stimulus-toolbox

# Install dependencies
pip install -r requirements.txt

# Run the generator
streamlit run dot_array_generator.py

# Or run the analyzer
streamlit run dot_array_analyzer.py
```

## Validation

The analyzer has been validated against 500 generated stimuli with known ground truth parameters:

| Parameter | Correlation (r) | Exact Match |
|-----------|----------------|-------------|
| Numerosity | > .99 | 99.8% |
| Cumulative Area | .99 | — |
| Average Element Size | .99 | — |
| Convex Hull Area | 1.00 | — |

See the accompanying paper for full validation details.

## Citation

If you use this toolbox in your research, please cite:

> Aulet, L.S. (2026). Dot Array Stimulus Toolbox: An Open-Source Solution for Generating and Analyzing Non-Symbolic Number Stimuli. _PsyArXiv_.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This toolbox was developed to provide an open-source, validated alternative to existing tools that require commercial software licenses. It aims to improve accessibility and reproducibility in numerical cognition research.
