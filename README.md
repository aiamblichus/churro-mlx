# Churro MLX

<img align="right" src="./vibecoded.png" width="200">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A fast, efficient MLX-based implementation of [Churro](https://github.com/stanford-oval/Churro), a specialized vision-language model for high-accuracy historical document OCR. This implementation uses Apple's MLX framework for optimized performance on Apple Silicon.

## About Churro

Churro is a 3B-parameter open-weight vision-language model (VLM) fine-tuned specifically for historical text recognition. It was trained on **Churro-DS**, the largest historical text recognition dataset to date, comprising 99,491 pages from 155 historical corpora spanning 22 centuries across 46 language clusters.

According to the [research paper](https://arxiv.org/abs/2509.19768), Churro achieves:

- **82.3%** normalized Levenshtein similarity on printed historical documents
- **70.1%** normalized Levenshtein similarity on handwritten historical documents
- Outperforms Gemini 2.5 Pro while being **15.5 times more cost-effective**

## Features

- 🚀 **Optimized for Apple Silicon** - Uses MLX for fast inference on Mac GPUs
- 📄 **Historical Document OCR** - Specialized for documents from the 3rd century BC to the 20th century
- 🌍 **Multilingual Support** - Handles 46 language clusters including historical variants and dead languages
- 📝 **XML Output** - Produces structured HistoricalDocument XML format
- 🎨 **Pretty Printing** - Optional syntax-highlighted output using Rich
- ⚡ **4-bit Quantization** - Efficient model size with minimal accuracy loss

## Installation

### Prerequisites

- Python 3.13 or higher
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

### Model Conversion

Before first use, you need to convert the Churro model to MLX format. This requires the original HuggingFace model:

```bash
uv run python -m mlx_vlm.convert \
  --hf-path stanford-oval/churro-3B \
  --mlx-path mlx_churro_4bit \
  -q \
  --q-bits 4
```

This will download the model from HuggingFace and convert it to a 4-bit quantized MLX format, saving it to the `mlx_churro_4bit` directory.

## Usage

### Basic Usage

Transcribe a single historical document image:

```bash
uv run python churro_cli.py --pretty --max-tokens 20000 examples/Figaro_1893.JPEG
```

### Command-Line Options

```bash
uv run python churro_cli.py [OPTIONS] IMAGE

Arguments:
  IMAGE                    Path to the historical document image

Options:
  -m, --model PATH         Path to MLX converted model directory
                           [default: mlx_churro_4bit]
  --max-tokens INTEGER     Max tokens to generate (increase for dense pages)
                           [default: 2000]
  -t, --temp FLOAT         Sampling temperature (0.0 for deterministic)
                           [default: 0.6]
  -v, --verbose            Print status messages to stderr
  -p, --pretty             Pretty print output with syntax highlighting using rich
  --help                   Show this message and exit
```

### Examples

**Basic transcription:**

```bash
uv run python churro_cli.py examples/Pascal_Response_du_provincial.JPEG
```

**With pretty-printed XML output:**

```bash
uv run python churro_cli.py examples/Figaro_1893.JPEG --pretty
```

**Verbose output with custom settings:**

```bash
uv run python churro_cli.py document.jpg \
  --max-tokens 4000 \
  --temp 0.3 \
  --verbose \
  --pretty
```

**Using a different model path:**

```bash
uv run python churro_cli.py document.jpg --model /path/to/custom/model
```

## Output Format

By default, Churro outputs transcriptions in **HistoricalDocument XML** format, a structured schema designed to capture complex layouts, scribal edits, and missing text while preserving reading order.

### Example Output

```xml
<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata>
    <Language>French</Language>
    <WritingDirection>ltr</WritingDirection>
    <PhysicalDescription>Single printed page from a book.</PhysicalDescription>
  </Metadata>
  <Page>
    <Header>
      <PageNumber>1</PageNumber>
      <Heading type="main">
        <Line>RESPONSE DV PROVINCIAL</Line>
      </Heading>
    </Header>
    <Body>
      <Paragraph>
        <Line>MONSIEVR,</Line>
        <Line>Vos deux lettres n'ont pas esté pour moy seul...</Line>
      </Paragraph>
    </Body>
  </Page>
</HistoricalDocument>
```

The `--pretty` flag enables syntax-highlighted XML output with line numbers for better readability.

## Model Details

- **Base Model**: Qwen 2.5 VL (3B parameters)
- **Quantization**: 4-bit
- **Max Image Dimension**: 2500 pixels (images are automatically resized)
- **Supported Scripts**: Latin, Cyrillic, Greek, Hebrew, Arabic, Devanagari, Bengali, Khmer, Japanese, Chinese, and more

## Performance Tips

1. **Dense Pages**: Increase `--max-tokens` (e.g., 4000-6000) for pages with a lot of text
2. **Deterministic Output**: Use `--temp 0.0` for reproducible results
3. **Large Images**: Images are automatically resized to fit within 2500×2500 pixels while preserving aspect ratio
4. **GPU Acceleration**: MLX automatically uses Apple Silicon GPUs when available

## Limitations

- Currently optimized for Apple Silicon (M-series chips)
- Model conversion required before first use
- Best performance on documents from the training period (3rd century BC - 20th century)
- Some languages may have lower accuracy if underrepresented in the training data

## Citation

If you use Churro in your research, please cite the original paper:

```bibtex
@inproceedings{semnani2025churro,
  title        = {{CHURRO}: Making History Readable with an Open-Weight Large Vision-Language Model for High-Accuracy, Low-Cost Historical Text Recognition},
  author       = {Semnani, Sina J. and Zhang, Han and He, Xinyan and Tekg{\"u}rler, Merve and Lam, Monica S.},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
  year         = {2025}
}
```

## Related Projects

- **[Original Churro Repository](https://github.com/stanford-oval/Churro)** - PyTorch implementation with full training and evaluation code
- **[Churro-DS Dataset](https://github.com/stanford-oval/Churro)** - The historical document dataset used for training
- **[MLX VLM](https://github.com/ml-explore/mlx-examples/tree/main/vlm)** - MLX vision-language model framework

## License

- **Model Weights**: Qwen research license (see [HuggingFace model card](https://huggingface.co/stanford-oval/churro-3B))
- **Dataset**: Research purposes only (see original repository for details)
- **Code**: Apache 2.0

## Acknowledgments

This MLX implementation is based on the original Churro model developed by the Stanford OVAL lab. Special thanks to the Churro team for their groundbreaking work on historical document OCR.

---

For more information about Churro, visit the [original repository](https://github.com/stanford-oval/Churro) or read the [research paper](https://arxiv.org/abs/2509.19768).
