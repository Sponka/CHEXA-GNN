import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, RunSamplesPaths
from src.inference import run_preprocessed_samples


def main():
    run_cfg = RunSamplesPaths()
    model_cfg = ModelConfig()
    output_dir = run_preprocessed_samples(run_cfg=run_cfg, model_cfg=model_cfg, batch_size=8)
    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()
