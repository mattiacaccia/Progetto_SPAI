import yaml
from pathlib import Path

def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Configurazione caricata con successo:")
    print(f"Progetto: {cfg['project']['name']}")
    print(f"Device selezionato: {cfg['project']['device']}")
    print(f"Batch Size: {cfg['dataset']['batch_size']}")