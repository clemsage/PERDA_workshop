#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# Commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --job-name=semantic_segmentation
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time=02:00:00
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user clement.sage@univ-ubs.fr
#SBATCH --output=output/%x-%j.txt

 # Exécution du script
uv run semantic_segmentation.py --max-epochs 50 --pre-trained-weights ResNet18_Weights.SENTINEL2_ALL_MOCO
