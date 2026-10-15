#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:2
#SBATCH --constraint 2080
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user clement.sage@univ-ubs.fr

# Si besoin, spécification de la version CUDA mise à disposition sur le cluster
# setcuda 12.2

# Exécution du script habituellement utilisé, on utilise la variable CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservés (toujours à partir de 0)

uv run semantic_segmentation.py
