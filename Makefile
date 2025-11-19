convert:
	uv run python -m mlx_vlm.convert \
          --hf-path stanford-oval/churro-3B \
          --mlx-path mlx_churro_4bit \
          -q \
          --q-bits 4