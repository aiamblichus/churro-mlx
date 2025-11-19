import sys
from pathlib import Path

import typer
from PIL import Image
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer()

# Configuration constants from the original implementation
DEFAULT_MODEL_PATH = "mlx_churro_4bit"  # Points to the converted weights
DEFAULT_SYSTEM_MESSAGE = (
    "Transcribe the entiretly of this historical documents to XML format."
)
MAX_IMAGE_DIM = 2500  # Churro specific resizing rule


def resize_image_to_fit(image: Image.Image, max_dim: int) -> Image.Image:
    """
    Resizes image to ensure longest side is <= max_dim, matching Churro's logic.
    """
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image

    scale = min(max_dim / width, max_dim / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))

    # Use LANCZOS for high-quality downsampling
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def run_inference(
    image_path: Path,
    model_path: str,
    max_tokens: int,
    temperature: float,
    verbose: bool,
    pretty: bool = False,
):
    # 1. Load Model
    if verbose:
        print(f"Loading model from {model_path}...", file=sys.stderr)

    try:
        model, processor = load(model_path)
        config = load_config(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print(
            "Did you run the conversion step? (python -m mlx_vlm.convert ...)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Prepare Image
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(image_path)
        img = resize_image_to_fit(img, MAX_IMAGE_DIM)
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Construct Prompt (Qwen 2.5 VL Chat Template)
    # Churro was trained with a specific system prompt.
    conversation = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe this."},
            ],
        },
    ]

    # 4. Apply Chat Template
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    if verbose:
        print("Generating transcription...", file=sys.stderr)

    # 5. Generate
    output = generate(
        model,
        processor,
        prompt,
        [img],
        max_tokens=max_tokens,
        verbose=False,  # We handle printing manually
        temp=temperature,
    )

    # 6. Output
    # Handle both GenerationResult objects and plain strings
    if hasattr(output, "text"):
        text_output = output.text
    else:
        text_output = str(output)

    if pretty:
        console = Console()
        # Use rich Syntax to highlight XML
        syntax = Syntax(text_output, "xml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        print(text_output)


@app.command()
def main(
    image: Path = typer.Argument(..., help="Path to the historical document image"),
    model: str = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model",
        "-m",
        help="Path to MLX converted model directory",
    ),
    max_tokens: int = typer.Option(
        2000, "--max-tokens", help="Max tokens to generate (increase for dense pages)"
    ),
    temp: float = typer.Option(
        0.6, "--temp", "-t", help="Sampling temperature (0.0 for deterministic)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print status messages to stderr"
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty",
        "-p",
        help="Pretty print output with syntax highlighting using rich",
    ),
):
    """
    Churro MLX: Historical Document OCR
    """
    run_inference(image, model, max_tokens, temp, verbose, pretty)


if __name__ == "__main__":
    app()
