# Multi-Agent Adaptive Cyber Deception

## Objectif

Ce projet utilise le Reinforcement Learning (RL) pour apprendre à placer des leurres (honeypots) dans un réseau afin de tromper les attaquants.

Deux approches sont étudiées :
- **C1 (Centralized)** : un seul agent gère le placement des leurres dans tout le réseau.
- **M2 (Multi-Agent)** : plusieurs agents gèrent le placement des leurres dans différentes zones du réseau.

L'objectif est de comparer leurs performances.

## Structure du projet

```text
projetHautdeb/
│
├── baselines.py
├── evaluate.py
├── requirements.txt
├── sim.py
├── train.py
│
└── results/
    ├── models/
    │   ├── c1_z4_s0.zip
    │   ├── c1_z4_s1.zip
    │   ├── c1_z4_s2.zip
    │   ├── m2_z4_s0.zip
    │   ├── m2_z4_s1.zip
    │   └── m2_z4_s2.zip
    │
    └── rq1_z4_b3.csv
```

## 1. Installation (une seule fois)

Prérequis : Python 3.10 ou 3.11 et VS Code.

Ouvrir le dossier du projet dans VS Code, puis ouvrir un terminal et exécuter :

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Sur Mac/Linux, utiliser `source venv/bin/activate` à la place de la deuxième commande.

## 2. Comprendre les seeds

Une seed (graine) est une valeur qui initialise les éléments aléatoires de l'entraînement.

Chaque seed correspond à deux entraînements :
- Un pour C1.
- Un pour M2.

Nous utilisons 5 seeds au total : 0, 1, 2, 3 et 4.

Les seeds 0, 1 et 2 sont déjà effectuées.

**Il reste donc à effectuer les seeds 3 et 4, soit 4 entraînements.**

## 3. Entraînements à effectuer

Chaque entraînement doit utiliser **300 000 steps**.

Dans le terminal, exécuter les commandes suivantes dans cet ordre :

**Seed 3 :**

```bash
python train.py --mode c1 --seed 3 --steps 300000
python train.py --mode m2 --seed 3 --steps 300000
```

**Seed 4 :**

```bash
python train.py --mode c1 --seed 4 --steps 300000
python train.py --mode m2 --seed 4 --steps 300000
```

Chaque entraînement sauvegarde son modèle dans `results/models/`.

À la fin, les quatre fichiers suivants doivent être présents :

```text
c1_z4_s3.zip
m2_z4_s3.zip
c1_z4_s4.zip
m2_z4_s4.zip
```

**Important :** ne pas modifier les seeds, le nombre de steps ou les paramètres d'entraînement.

## 4. Évaluation

Une fois les quatre entraînements terminés, exécuter :

```bash
python evaluate.py
```

Ce script permet d'évaluer les modèles et de comparer leurs performances avec les baselines.

## 5. Résultats attendus

À la fin, le dossier `results/models/` doit contenir 10 modèles :

| Approche | Seeds | Nombre de modèles |
|---|---|---:|
| C1 | 0, 1, 2, 3, 4 | 5 |
| M2 | 0, 1, 2, 3, 4 | 5 |
| **Total** | | **10** |

Conserver les fichiers déjà présents dans `results/`. Ne pas les supprimer ni les remplacer.

## 6. En cas de problème

En cas d'erreur pendant l'installation ou l'entraînement, conserver le message d'erreur complet du terminal pour pouvoir identifier le problème.
