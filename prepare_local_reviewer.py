"""Download and verify the optional local Qwen document-review model."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--check-only", action="store_true", help="只检查依赖和本地缓存"
    )
    args = parser.parse_args()

    try:
        import torch
        import transformers
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        from huggingface_hub import snapshot_download
    except ImportError as error:
        print(f"缺少审校器依赖：{error}")
        print("请先运行：python -m pip install -r requirements-reviewer.txt")
        return 2

    if version_tuple(transformers.__version__) < (4, 51):
        print(f"transformers {transformers.__version__} 过旧，需要 4.51 或更高版本。")
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA；4-bit 审校器要求 NVIDIA GPU。")
        return 3
    properties = torch.cuda.get_device_properties(0)
    memory_gib = properties.total_memory / 1024**3
    print(f"GPU：{properties.name}，显存 {memory_gib:.1f} GiB")
    if memory_gib < 7.5:
        print("警告：显存低于 7.5 GiB，建议降低 max_context_tokens。")

    try:
        location = snapshot_download(
            args.model,
            local_files_only=args.check_only,
        )
    except Exception as error:
        action = "本地缓存检查" if args.check_only else "模型下载"
        print(f"{action}失败：{type(error).__name__}: {error}")
        return 4

    versions = {
        package: importlib.metadata.version(package)
        for package in ("transformers", "accelerate", "bitsandbytes")
    }
    print(f"审校模型已就绪：{location}")
    print("依赖版本：" + "，".join(f"{key}={value}" for key, value in versions.items()))
    print("现在运行 start_accurate_translator.cmd，并勾选“自动深度审校”。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
