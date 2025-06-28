#!/usr/bin/env python3
# Script pour utiliser un modèle GGUF local avec MLX
# Utilisation : python use_local_model.py --gguf /path/to/model.gguf --prompt "Votre prompt"

import argparse
import time
from pathlib import Path

import mlx.core as mx
import models


def main():
    parser = argparse.ArgumentParser(description="Utiliser un modèle GGUF local avec MLX")
    parser.add_argument(
        "--gguf",
        type=str,
        required=True,
        help="Chemin vers le fichier GGUF local"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Bonjour, comment ça va ?",
        help="Le prompt à traiter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Nombre maximum de tokens à générer"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Température d'échantillonnage"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Graine aléatoire"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not Path(args.gguf).exists():
        print(f"❌ Erreur: Le fichier {args.gguf} n'existe pas")
        return
    
    print(f"🔄 Chargement du modèle depuis : {args.gguf}")
    
    # Configurer la graine aléatoire
    mx.random.seed(args.seed)
    
    # Charger le modèle (sans repo car le fichier existe déjà)
    model, tokenizer = models.load(args.gguf, repo=None)
    
    print(f"\n📝 Prompt: {args.prompt}")
    print("🤖 Réponse: ", end="", flush=True)
    
    # Encoder le prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    
    # Génération
    tic = time.time()
    tokens = []
    skip = 0
    
    for token, n in zip(
        models.generate(prompt_tokens, model, args.temp),
        range(args.max_tokens),
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
    
    print("\n" + "=" * 50)
    if len(tokens) > 0:
        prompt_tps = len(prompt_tokens) / prompt_time
        gen_tps = (len(tokens) - 1) / gen_time
        print(f"⚡ Prompt: {prompt_tps:.3f} tokens/sec")
        print(f"⚡ Génération: {gen_tps:.3f} tokens/sec")
        print(f"📊 Tokens générés: {len(tokens)}")
    else:
        print("❌ Aucun token généré pour ce prompt")


if __name__ == "__main__":
    main()
