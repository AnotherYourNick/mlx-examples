#!/usr/bin/env python3
# Script pour lister les modèles GGUF disponibles localement

import os
import subprocess
from pathlib import Path


def find_gguf_models():
    """Trouve tous les modèles GGUF disponibles localement"""
    models = []
    
    # 1. Chercher dans le cache Hugging Face
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        try:
            # Utiliser fd si disponible, sinon find
            result = subprocess.run(
                ["fd", r"\.gguf$", str(hf_cache)],
                capture_output=True,
                text=True,
                check=True
            )
            hf_models = result.stdout.strip().split('\n') if result.stdout.strip() else []
            for model in hf_models:
                if model:  # Éviter les lignes vides
                    name = Path(model).parent.parent.parent.name.replace("models--", "").replace("--", "/")
                    size = os.path.getsize(model) / (1024**3)  # Taille en GB
                    models.append({
                        "path": model,
                        "name": name,
                        "filename": Path(model).name,
                        "size_gb": f"{size:.1f} GB",
                        "source": "Hugging Face"
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback si fd n'est pas disponible
            try:
                result = subprocess.run(
                    ["find", str(hf_cache), "-name", "*.gguf", "-type", "f"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                hf_models = result.stdout.strip().split('\n') if result.stdout.strip() else []
                for model in hf_models:
                    if model:
                        name = Path(model).parent.parent.parent.name.replace("models--", "").replace("--", "/")
                        size = os.path.getsize(model) / (1024**3)
                        models.append({
                            "path": model,
                            "name": name,
                            "filename": Path(model).name,
                            "size_gb": f"{size:.1f} GB",
                            "source": "Hugging Face"
                        })
            except subprocess.CalledProcessError:
                pass
    
    # 2. Chercher dans le répertoire LM Studio
    lm_studio_cache = Path.home() / ".cache" / "lm-studio" / "models"
    if lm_studio_cache.exists():
        try:
            result = subprocess.run(
                ["fd", r"\.gguf$", str(lm_studio_cache)],
                capture_output=True,
                text=True,
                check=True
            )
            lm_models = result.stdout.strip().split('\n') if result.stdout.strip() else []
            for model in lm_models:
                if model:
                    size = os.path.getsize(model) / (1024**3)
                    models.append({
                        "path": model,
                        "name": Path(model).parent.name,
                        "filename": Path(model).name,
                        "size_gb": f"{size:.1f} GB",
                        "source": "LM Studio"
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return models


def print_models(models):
    """Affiche la liste des modèles de manière formatée"""
    if not models:
        print("❌ Aucun modèle GGUF trouvé localement")
        print("\n💡 Pour télécharger un modèle :")
        print("   • Via LM Studio : Lancez l'app et téléchargez un modèle")
        print("   • Via MLX : python generate.py --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        return
    
    print(f"🎯 {len(models)} modèle(s) GGUF trouvé(s) localement :\n")
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. 📁 {model['name']}")
        print(f"    📄 Fichier: {model['filename']}")
        print(f"    💾 Taille: {model['size_gb']}")
        print(f"    🏷️  Source: {model['source']}")
        print(f"    📍 Chemin: {model['path']}")
        print()
    
    print("🚀 Pour utiliser un modèle avec MLX :")
    print("   python use_local_model.py --gguf \"<chemin_vers_le_modèle>\" --prompt \"Votre prompt\"")
    print("\n📖 Exemple :")
    if models:
        print(f"   python use_local_model.py --gguf \"{models[0]['path']}\" --prompt \"Bonjour !\"")


def main():
    print("🔍 Recherche des modèles GGUF locaux...\n")
    models = find_gguf_models()
    print_models(models)


if __name__ == "__main__":
    main()
