from fonction import read_fasta_file

seq = input("Enter ARN sequence or a fasta file path : ")
scoring_function = input("Select scoring function : \n1. RASP \n2. DFIRE-RNA \n3. rsRNASP\n")
match scoring_function:
    case "1":
        print('RASP')
    case "2":
        print('DFIRE-RNA')
    case "3":
        print('rsRNASP')
    case _:
        raise ValueError("Invalid scoring function")

if seq.endswith(".fasta"):
    seq = read_fasta_file(seq)
    
