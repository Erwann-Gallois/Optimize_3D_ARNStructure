import multiprocessing as mp
import torch
import itertools
from tqdm import tqdm
import os
from parse_rasp_potentials import load_rasp_potentials
from classe.RNA_RASP_Rigid_parallel import RNA_RASP_Rigid_parallel
from fonction import generer_arn_droit

def run_task(args):
    params, pdb_path, potential_tensor = args
    # params = (lr, epochs, n_cycles, noise_c, noise_a)
    filename = f"output_lr{params[0]}_ep{params[1]}_cy{params[2]}_nc{params[3]}_na{params[4]}.pdb"
    output_dir = "fichier_arn/resultats_grid"
    os.makedirs(output_dir, exist_ok=True)
    model = RNA_RASP_Rigid_parallel(
        pdb_path=pdb_path,
        potential_tensor=potential_tensor,
        lr=params[0],
        epochs_per_cycle=params[1],
        num_cycles=params[2],
        noise_coords=params[3],
        noise_angles=params[4],
        verbose=False,
        output_path=None,
        backbone_weight=params[5]
    )
    model.run_optimization()
    rg_final = model.get_compactness()
    return (*params, model.best_score, rg_final)

if __name__ == "__main__":
    # 1. Charger les potentiels UNE SEULE FOIS
    print("Chargement des potentiels...")
    taille, dict_pots = load_rasp_potentials("potentials/all.nrg")
    # Création du tenseur global
    pot_tensor = torch.zeros(taille)
    for (k, t1, t2, dist), energy in dict_pots.items():
        pot_tensor[k, t1, t2, dist] = energy
        pot_tensor[k, t2, t1, dist] = energy
    pot_tensor.share_memory_() # Partage efficace entre processus

    # 2. Définir la grille
    lrs = [0.05, 0.1, 0.2, 0.3]
    epochs = [100, 150, 200]
    cycles = [50, 100, 150, 200]
    backbone_weight = [50, 100, 500, 1000]
    noises_c = [1.0, 2.0, 3.0]
    noises_a = [0.3, 0.6, 0.8]
    
    sequence_arn = "GGAACCGGUGCGCAUAACCACCUCAGUGCGAGCAA"
    pdb_initial = "fichier_arn/mon_arn_rigid_droit.pdb"
    generer_arn_droit(sequence_arn, pdb_initial)

    combinations = list(itertools.product(lrs, epochs, cycles, noises_c, noises_a, backbone_weight))
    tasks = [(c, pdb_initial, pot_tensor) for c in combinations]

    # 3. Exécuter sur 50 cœurs
    print(f"Lancement du Grid Search : {len(combinations)} combinaisons.")
    with mp.Pool(processes=48) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(run_task, tasks), total=len(tasks), desc="Optimisation"):
            results.append(result)

    # 4. Trier et afficher
    results.sort(key=lambda x: x[-1])
    compact_results = [r for r in results if r[-1] < 20.0] 
    # On trie ensuite ces structures compactes par SCORE RASP (le plus bas est le mieux)
    compact_results.sort(key=lambda x: x[-2]) 

    print("\n--- TOP 10 REPLIEMENTS (Compacts + Meilleur Score) ---")
    print("LR | Ep | Cy | NC | NA | BBW | SCORE | Rg")
    for r in compact_results[:10]:
        print(f"{r[0]:.2f} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.1f} | {r[5]} | {r[-2]:.2f} | {r[-1]:.2f}")