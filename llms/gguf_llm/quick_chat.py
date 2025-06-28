#!/usr/bin/env python3
# Raccourci pour lancer rapidement un chat avec le premier modèle disponible

import subprocess
import sys
from pathlib import Path


def main():
    # Trouver le premier modèle GGUF disponible
    try:
        result = subprocess.run(
            ["fd", r"\.gguf$", str(Path.home() / ".cache" / "huggingface")],
            capture_output=True,
            text=True,
            check=True
        )
        models = result.stdout.strip().split('\n')
        
        if not models or not models[0]:
            print("❌ Aucun modèle GGUF trouvé")
            print("💡 Téléchargez d'abord un modèle avec :")
            print("   python generate.py --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
            return
        
        model_path = models[0]
        model_name = Path(model_path).name
        
        print(f"🚀 Démarrage rapide avec {model_name}")
        print("=" * 50)
        
        # Lancer le chat avec le modèle trouvé
        subprocess.run([
            sys.executable, "chat.py", 
            "--model", model_path, 
            "--temp", "0.1"
        ])
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Erreur lors de la recherche des modèles")
        print("💡 Assurez-vous d'avoir téléchargé au moins un modèle GGUF")


if __name__ == "__main__":
    main()
