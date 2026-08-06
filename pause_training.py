"""Request a safe pause after the current optimizer step."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    pointer = PROJECT_DIR / "output" / "training.current.pause"
    if not pointer.is_file():
        raise SystemExit("No active training control file was found.")
    pause_request = Path(pointer.read_text(encoding="utf-8").strip())
    pause_request.parent.mkdir(exist_ok=True, parents=True)
    pause_request.touch()
    print(f"Pause requested: {pause_request}")
    print("Training will save last_training_state.pt and exit after the current step.")


if __name__ == "__main__":
    main()
