# Dependencies

Recommended to install [uv][uv-link] to manage dependencies. This project also
requires [PyTorch][pytorch-link] and [Timm][timm-link], which should be installed
using pip through uv. This would look something like:

```bash
uv pip install torch torchvision --index-url [check-link-for-correct-url-for-hardware]
uv pip install timm
```

[uv-link]: https://docs.astral.sh/uv/getting-started/installation/
[pytorch-link]: https://pytorch.org/get-started/locally/
[timm-link]: https://huggingface.co/docs/timm/installation
