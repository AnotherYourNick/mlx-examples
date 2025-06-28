#!/usr/bin/env python3
# Chat interactif avec les modèles GGUF locaux

import argparse
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import models


def find_gguf_models():
    """Trouve tous les modèles GGUF disponibles localement"""
    models_list = []
    
    # Chercher dans le cache Hugging Face
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        try:
            result = subprocess.run(
                ["fd", r"\.gguf$", str(hf_cache)],
                capture_output=True,
                text=True,
                check=True
            )
            hf_models = result.stdout.strip().split('\n') if result.stdout.strip() else []
            for model in hf_models:
                if model:
                    name = Path(model).parent.parent.parent.name.replace("models--", "").replace("--", "/")
                    size = os.path.getsize(model) / (1024**3)
                    models_list.append({
                        "path": model,
                        "name": name,
                        "filename": Path(model).name,
                        "size_gb": f"{size:.1f} GB",
                        "source": "Hugging Face"
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Chercher dans LM Studio
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
                    models_list.append({
                        "path": model,
                        "name": Path(model).parent.name,
                        "filename": Path(model).name,
                        "size_gb": f"{size:.1f} GB",
                        "source": "LM Studio"
                    })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return models_list


def select_model(models_list):
    """Permet à l'utilisateur de sélectionner un modèle"""
    if not models_list:
        print("❌ Aucun modèle GGUF trouvé localement")
        print("\n💡 Téléchargez un modèle via LM Studio ou avec :")
        print("   python generate.py --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        return None
    
    print(f"🎯 {len(models_list)} modèle(s) disponible(s) :")
    print()
    
    for i, model in enumerate(models_list, 1):
        print(f"{i:2d}. 📁 {model['name']}")
        print(f"    📄 {model['filename']} ({model['size_gb']}) - {model['source']}")
        print()
    
    while True:
        try:
            choice = input("Sélectionnez un modèle (numéro) : ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(models_list):
                return models_list[choice - 1]
            else:
                print(f"⚠️  Veuillez entrer un numéro entre 1 et {len(models_list)}")
        except ValueError:
            print("⚠️  Veuillez entrer un numéro valide")
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            return None


def generate_response(model, tokenizer, prompt, max_tokens=150, temp=0.0):
    """Génère une réponse avec le modèle"""
    prompt_tokens = tokenizer.encode(prompt)
    
    tic = time.time()
    tokens = []
    skip = 0
    
    for token, n in zip(
        models.generate(prompt_tokens, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
            
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
            
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
    
    if len(tokens) > 0:
        prompt_tps = len(prompt_tokens) / prompt_time
        gen_tps = (len(tokens) - 1) / gen_time
        print(f"\n⚡ {gen_tps:.1f} tokens/sec")
    
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Chat interactif avec MLX")
    parser.add_argument("--model", type=str, help="Chemin vers un modèle GGUF spécifique")
    parser.add_argument("--temp", type=float, default=0.1, help="Température (0.0-1.0)")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum de tokens")
    
    args = parser.parse_args()
    
    print("🚀 MLX Chat - Interface pour modèles GGUF locaux")
    print("=" * 50)
    
    # Sélection du modèle
    if args.model and Path(args.model).exists():
        model_path = args.model
        print(f"📁 Utilisation du modèle : {Path(model_path).name}")
    else:
        print("🔍 Recherche des modèles disponibles...")
        models_list = find_gguf_models()
        selected_model = select_model(models_list)
        
        if not selected_model:
            return
        
        model_path = selected_model["path"]
        print(f"✅ Modèle sélectionné : {selected_model['name']}")
    
    # Chargement du modèle
    print(f"\n🔄 Chargement en cours...")
    mx.random.seed(42)
    
    try:
        model, tokenizer = models.load(model_path, repo=None)
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return
    
    # Chat interactif
    print(f"\n💬 Chat démarré ! (tapez 'quit' pour quitter)")
    print(f"⚙️  Température: {args.temp}, Max tokens: {args.max_tokens}")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n🤔 Vous: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break
            
            if not prompt:
                continue
            
            print("🤖 IA: ", end="", flush=True)
            generate_response(model, tokenizer, prompt, args.max_tokens, args.temp)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")


if __name__ == "__main__":
    main()
