from pathlib import Path
import yaml
from cerebro.config.model import Model

DEFAULT_PATH = Path(__file__).parent / "models.yaml"

def load_models(path: Path | str = DEFAULT_PATH) -> dict[str, Model]:
    raw = Path(path).read_text()
    data = yaml.safe_load(raw)

    return {
        name: Model(**model_data)
        for name, model_data in data["models"].items()
    }