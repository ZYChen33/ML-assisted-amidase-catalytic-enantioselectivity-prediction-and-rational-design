import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

def read_file(filename):
    f=open(filename,"r")
    lines=f.readlines()
    f.close()
    data={}
    target=[]
    elements={}
    index=[-1,]
    molecular=[]   
    for i in range(len(lines)):
        if lines[i].strip().split()[0]=="END":
            index.append(i)          #the index of END
    
    for j in range(len(index)):
        element=[]
        atom_position=[]
        if j==(len(index)-1):
            break
        else:
            for k in range(index[j]+2,(index[j+1]-1)):
                atom_position.append(np.array([float(lines[k].strip().split()[1]),float(lines[k].strip().split()[2]),
                                               float(lines[k].strip().split()[3])])) 
                
                element.append(lines[k].strip().split()[0])
        target.append(int(lines[index[j+1]-1].strip('\n')))
        molecular.append(lines[index[j]+1].strip('\n'))
        data[lines[index[j]+1].strip('\n')]=atom_position
        elements[lines[index[j]+1].strip('\n')]=element
    return molecular,data,target,elements   
 

############################################################################################################


MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

def Vocabulary(data):
    cset=set()
    for m in data:
        mol=MolTree(m)
        for c in mol.nodes:
            cset.add(c.smiles)
    return cset

class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x,y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
        
        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
            #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    #Compute Maximum Spanning Tree
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []

            
# Creating dictionary for vocab/categorical  
def Vocab2Cat(vocabset):
    vocab=list(vocabset)
    chars=list(np.arange(len(vocabset)))
    MolDict=dict(zip(vocab,chars))
    return MolDict

# Obtaining the clusters for moles in the training set 
def Clusters(data):
    clusters=[]
    for m in data:
        c=[] #using c for clusters
        tree=MolTree(m)
        for node in tree.nodes:
            c.append(node.smiles)
        clusters.append(c)
    return clusters

# Turning each set of clusters for each molecule into categorical labels
def Cluster2Cat(clusters,MolDict):
    cat=[]
    for cluster in clusters:
        l=[]
        for c in cluster:
            l.append(MolDict[c])
        cat.append(l)
    return cat

# Creating vector descriptions from one hot encoded labels of clusters
# size is the number of categorical labels
def Vectorize(catdata,size):
    vectors=[]
    for c in catdata:
        c0=np.array(c).astype(int)
        b=np.zeros((len(c0),size))
        b[np.arange(len(c0)),c0]=1
        b1=b.sum(axis=0)
        vectors.append(b1)
    return vectors

#Try rings first: Speed-Up 
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
    all_attach_confs = []
    singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach(node.mol, neighbors[:depth+1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if len(candidates) == 0:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles:
            continue
        cand_smiles.add(smiles)
        Chem.Kekulize(cand_mol)
        candidates.append( (smiles,cand_mol,amap) )

    return candidates

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

        
def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_decomposition(mol_smiles, outputs= None, output_name = 'output'):
    # Generating Cliques 
    vocab=Vocabulary(mol_smiles)
    size=len(vocab)
    vocabl = list(vocab)
    MolDict=Vocab2Cat(vocab)
    clustersTR=Clusters(mol_smiles)
    catTR=Cluster2Cat(clustersTR,MolDict)
    clique_decomposition=Vectorize(catTR,size)

    descriptors_df = pd.DataFrame(data=clique_decomposition, columns = [x for x in range(len(vocabl))])

    if (outputs != None):
        descriptors_df[output_name] = output

    return descriptors_df, vocabl

def get_data_smiles(fileneme):
    with open(fileneme,'r') as f:
        lines=f.readlines()
    molecular=[]
    for i in range(len(lines)):
        molecular.append(lines[i].strip('\n'))
    return molecular

if __name__ == "__main__":
    with open("jtnn.csv", "r") as file:
        jtnn_std=pd.read_csv(file).iloc[:,1:]  #as reference
    jtnn_std_col=jtnn_std.columns
    
    new_mole=get_data_smiles("amidase_trial_smiles.txt")   #only smiles
    new_jtnn_df,new_vocabl=get_clique_decomposition(new_mole)
    new_jtnn=np.array(new_jtnn_df)
    jtnn_array=np.zeros((len(new_mole),len(jtnn_std_col)))
    for fi in range(len(jtnn_std_col)):
        for fj in range(len(new_vocabl)):
            if jtnn_std_col[fi]==new_vocabl[fj]:
                for natoms in range(len(new_mole)):
                    jtnn_array[natoms][fi]=new_jtnn[natoms][fj]
    newdata_jtnn_df=pd.DataFrame(jtnn_array)
    newdata_jtnn_df.columns=jtnn_std_col
    newdata_jtnn_df.to_csv('amidase_jtnn_trial.csv')
    
with open("amidase_sf_trial.csv", "r") as file:  
    sf=pd.read_csv(file).iloc[:,1:]    
sf_col=list(sf.columns)

with open("amidase_jtnn_trial.csv", "r") as file: 
    jtnn=pd.read_csv(file).iloc[:,1:]
jtnn_col=list(jtnn.columns)
sf_jtnn=pd.concat([sf,jtnn],axis=1,ignore_index=True)
sfjtnn_col=sf_col+jtnn_col
sf_jtnn.columns=sfjtnn_col
sf_jtnn.to_csv('amidase_jtnnsf_trial.csv')
