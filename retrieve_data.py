import os
import requests
import time
from Bio.PDB import PDBList
import argparse
from scipy import stats

def main():
    parser = argparse.ArgumentParser(description="CLI interface for extracting atom distances from RNA structures.")
    parser.add_argument("-f", "--folder_path", help="Path to the folder containing CIF files.")
    parser.add_argument("-r", "--resolution", default=1.1, type=float, help="Maximum resolution for filtering structures.")
    parser.add_argument("-p", "--page", default=100, type=int, help="Number of structures to retrieve per page.")
    parser.add_argument("-e", "--extension", default="mmCif", choices=["mmCif", "pdb"], help="File format to download (mmCif or pdb).")
    args = parser.parse_args()
    ids = chercher_tous_rna_ids(args.resolution, args.page)
    if ids:
        telecharger_si_absent(ids, args.folder_path, args.extension, need_sleep=False)


# --- CONFIGURATION ---
def chercher_tous_rna_ids(res_limit, page):
    """Cherche TOUS les IDs en gérant la pagination de l'API RCSB."""
    print(f"--- Recherche de toutes les structures (Résolution < {res_limit}Å) ---")
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    all_ids = []
    start = 0
    total_count = 1 

    while start < total_count:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less",
                            "value": res_limit
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "struct_keywords.pdbx_keywords",
                            "operator": "contains_phrase",
                            "value": "RNA"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": page},
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "score", "direction": "desc"}]
            }
        }

        response = requests.post(url, json=query)
        if response.status_code == 200:
            data = response.json()
            total_count = data.get('total_count', 0)
            ids = [res['identifier'] for res in data.get('result_set', [])]
            all_ids.extend(ids)
            print(f"[{len(all_ids)}/{total_count}] Identifiants récupérés...")
            start += page
        else:
            print(f"Erreur API : {response.status_code}")
            break
    return all_ids

def telecharger_si_absent(pdb_ids, folder, file_format, need_sleep=True):
    """Télécharge les fichiers uniquement s'ils ne sont pas déjà présents."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    pdbl = PDBList()
    extensions = {"mmCif": ".cif", "pdb": ".pdb"}
    ext = extensions.get(file_format, ".cif")
    
    deja_presents = 0
    a_telecharger = []

    # 1. Analyse préalable du dossier
    for pdb_id in pdb_ids:
        # Biopython nomme souvent les fichiers en minuscule : 1abc.cif
        filename = f"{pdb_id.lower()}{ext}"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            deja_presents += 1
        else:
            a_telecharger.append(pdb_id)

    print(f"\nStatistiques du dossier :")
    print(f"  - Déjà présents : {deja_presents}")
    print(f"  - À télécharger : {len(a_telecharger)}")

    # 2. Téléchargement effectif
    if not a_telecharger:
        print("Tout est déjà à jour !")
        return

    for i, pdb_id in enumerate(a_telecharger):
        print(f"[{i+1}/{len(a_telecharger)}] Téléchargement de {pdb_id}...")
        # flat=True évite la création de sous-répertoires bizarres
        pdbl.retrieve_pdb_file(pdb_id, pdir=folder, file_format=file_format)
        if need_sleep:
            time.sleep(1)



# --- EXÉCUTION ---
if __name__ == "__main__":
    main()