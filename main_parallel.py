import multiprocessing as mp
import torch
import itertools
from parse_rasp_potentials import load_rasp_potentials
from classe.RNA_RASP_Rigid_parallel import RNA_RASP_Rigid_parallel
from fonction import generer_arn_droit

def run_task(args):
    params, pdb_path, potential_tensor = args
    # params = (lr, epochs, n_cycles, noise_c, noise_a)
    
    model = RNA_RASP_Rigid_parallel(
        pdb_path=pdb_path,
        potential_tensor=potential_tensor,
        lr=params[0],
        epochs_per_cycle=params[1],
        num_cycles=params[2],
        noise_coords=params[3],
        noise_angles=params[4],
        verbose=False
    )
    model.run_optimization()
    return (*params, model.best_score)

if __name__ == "__main__":
    # 1. Charger les potentiels UNE SEULE FOIS
    print("Chargement des potentiels...")
    taille, dict_pots = load_rasp_potentials("potentials/c3.nrg")
    # Création du tenseur global
    pot_tensor = torch.zeros(taille)
    for (k, t1, t2, dist), energy in dict_pots.items():
        pot_tensor[k, t1, t2, dist] = energy
        pot_tensor[k, t2, t1, dist] = energy
    pot_tensor.share_memory_() # Partage efficace entre processus

    # 2. Définir la grille
    lrs = [0.1, 0.2, 0.3]
    epochs = [500, 800]
    cycles = [50, 100]
    noises_c = [1.0, 2.0, 3.0]
    noises_a = [0.3, 0.6, 0.8]
    
    sequence_arn = "GGAACCGGUGCGCAUAACCACCUCAGUGCGAGCAA"
    pdb_initial = "fichier_arn/mon_arn_rigid_droit.pdb"
    generer_arn_droit(sequence_arn, pdb_initial)

    combinations = list(itertools.product(lrs, epochs, cycles, noises_c, noises_a))
    tasks = [(c, pdb_initial, pot_tensor) for c in combinations]

    # 3. Exécuter sur 50 cœurs
    print(f"Lancement du Grid Search : {len(combinations)} combinaisons.")
    with mp.Pool(processes=18) as pool:
        results = pool.map(run_task, tasks)

    # 4. Trier et afficher
    results.sort(key=lambda x: x[-1])
    print("\nMeilleurs paramètres (LR, Epochs, Cycles, NoiseC, NoiseA | Score) :")
    for r in results[:10]:
        print(r)