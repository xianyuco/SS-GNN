from rdkit import Chem
import pickle
from utils import get_gnn_features
import os


threshholds = [3, 4, 5, 6, 7, 8, 9, 10]
coreset_path = "/media/ps/hd6/netblind/viewpro/data/coreset/"
coreset_names = os.listdir(coreset_path)

for thr in threshholds:
    target_path = "/media/ps/ssd3/netblind/scoredata/mice_features/"
    root = '/media/ps/hd6/netblind/viewpro/data/pdbbind/alldata/'
    target_path = os.path.join(target_path, "all_"+str(thr)+"A")
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    exitsted_datas = os.listdir(target_path)
    exitsted_datas = [esd.split(".")[0] for esd in exitsted_datas]
    n_train = 0
    m_train = 0
    exeptions_all = []
    allDatas = os.listdir(root)
    for i_all in allDatas:
        if i_all in coreset_names:
            if os.path.exists(os.path.join(target_path, i_all+".pkl")):
                os.remove(os.path.join(target_path, i_all+".pkl"))
            continue
        if i_all in exitsted_datas:
            continue
        else:
            n_train += 1
            pdb_path_all = os.path.join(root, i_all, i_all+'_protein.pdb')
            mol2_path_all = os.path.join(root, i_all, i_all+'_ligand.mol2')

            protein = Chem.MolFromPDBFile(pdb_path_all)
            if protein is None:
                protein = Chem.MolFromPDBFile(os.path.join(root, i_all, i_all+"_pocket.pdb"))
            ligand = Chem.MolFromMol2File(mol2_path_all)
            if ligand is None:
                suppl = Chem.SDMolSupplier(os.path.join(root, i_all, i_all+'_ligand.sdf'))
                mols = [mol for mol in suppl if mol]
                if len(mols)>0:
                    ligand = mols[0]
            if protein is None or ligand is None:
                m_train += 1
                print("protein or ligand is none")
                continue
            try:
                x_all, edge_index_all, edge_attr_all = get_gnn_features(protein, ligand, threshhold=thr)
                # label_train = dic_train[i_train]
                if x_all is None or edge_attr_all is None or edge_index_all is None:
                    m_train += 1
                    print("feature is none")
                    continue
                with open(os.path.join(target_path, i_all+'.pkl'), 'wb') as f_train2:
                    pickle.dump((x_all, edge_index_all, edge_attr_all), f_train2)
            except Exception as e:
                # print(e)
                print("exception")
                m_train += 1
                exeptions_all.append(i_all)
            print('all data is ', m_train, '/', n_train)

    print(exeptions_all)
