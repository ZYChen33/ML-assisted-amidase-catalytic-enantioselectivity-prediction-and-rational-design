
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Rc=6
lamda_list=[1,-1]
atomic_number={'H':1,'C':6,'N':7,'O':8,'F':9,'Cl':17,'Br':35}

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

def calc_parameters(N):
    mu_list=[]
    delta_r=(Rc-1.5)/(N-1)
    eta=1/(2*np.square(delta_r))
    for n in range(N):
        mu_n=0.5+n*delta_r
        mu_list.append(mu_n)
    return eta,mu_list

def calc_rad_SF(Nrad,molecular,data,elements):
    G_rad={}
    for k in molecular:
        num_atom=len(data[k])  
        atom=data[k]         #atom coordinate
        eta,mu_list=calc_parameters(Nrad)
        rad_atom_SFs=[]
        
        for i in range(num_atom):
            rad_atom_list=[]
            for mu in mu_list:
                radial_SF_i=0
                for j in range(num_atom):
                    if i!=j:
                        
                        Zj=atomic_number[elements[k][j]]
                        bond_i_j=np.sqrt(np.sum(np.square(atom[j]-atom[i])))
                        if bond_i_j<=Rc:
                            radial_SF_ij=np.exp(-eta*np.square(bond_i_j-mu))*(np.cos(np.pi*bond_i_j/Rc)+1)/2*Zj
                        else:
                            radial_SF_ij=0
                        radial_SF_i+=radial_SF_ij    
                rad_atom_list.append(radial_SF_i)
            rad_atom_SFs.append(rad_atom_list)
        G_rad[k]=rad_atom_SFs
    return G_rad

def calc_angular_atom_SFs(m,Nang,data,elements):
    num_atom=len(data[m])
    atom=data[m]        #atom coordinate,list
    angular_atom_SFs=[]
    eta,mu_list=calc_parameters(Nang)
    
    for i in range(num_atom):
        angular_SF_list=[]
        for lamda in lamda_list:
            for mu in mu_list:
                angular_SF_i=0               
                for j in range(num_atom):
                    for k in range(num_atom):
                        if i!=j and i!=k and k>j:
                            Zj=atomic_number[elements[m][j]]
                            Zk=atomic_number[elements[m][k]]
                            bond_j_i=np.sqrt(np.sum(np.square(atom[i]-atom[j])))
                            bond_k_i=np.sqrt(np.sum(np.square(atom[i]-atom[k])))
                            bond_j_k=np.sqrt(np.sum(np.square(atom[k]-atom[j])))
                            vector_j_i=np.array(atom[j]-atom[i])
                            vector_k_i=np.array(atom[k]-atom[i])
                            cos_angle_jik=vector_j_i.dot(vector_k_i)/(bond_j_i*bond_k_i)
                            if bond_j_i<=Rc and bond_k_i<=Rc and bond_j_k<=Rc:
                                angular_SF_jik=Zj*Zk*(1+lamda*cos_angle_jik)*np.exp(-eta*np.square(bond_j_i-mu))*np.exp(-eta*np.square(bond_k_i-mu))*np.exp(-eta*np.square(bond_j_k-mu))*(np.cos(np.pi*bond_j_i/Rc)+1)*(np.cos(np.pi*bond_k_i/Rc)+1)*(np.cos(np.pi*bond_j_k/Rc)+1)/8
                            else:
                                angular_SF_jik =0
                            angular_SF_i+=angular_SF_jik   
                
                angular_SF_list.append(angular_SF_i)
        angular_atom_SFs.append(angular_SF_list)
    return angular_atom_SFs
        
def calc_angular_mole_SFs(Nang,molecular,data,elements):
    G_angle={} 
    for m in molecular:
        G_angle[m]=calc_angular_atom_SFs(m,Nang,data,elements)
    return G_angle


def histogram_mole(m,G_rad,G_angle,Nrad,Nang,B):
    list_=[]
    rad_array=np.array(G_rad[m])
    angle_array=np.array(G_angle[m])
    
    rad_min=rad_array.min()
    rad_max=rad_array.max()
    angle_min=angle_array.min()
    angle_max=angle_array.max()
    for i in range(Nrad):
        hist_rad,_=np.histogram(rad_array[:,i],bins=B,range=(rad_min,rad_max))
        list_.extend(hist_rad)
    for j in range(Nang*2):
        hist_ang,_=np.histogram(angle_array[:,j],bins=B,range=(angle_min,angle_max))
        list_.extend(hist_ang)
    return np.array(list_)
        
def histogram_wACSFs(G_rad,G_angle,Nrad,Nang,B,molecular):
    hist_dict={}
    for m in molecular:
        hist_dict[m]=histogram_mole(m,G_rad,G_angle,Nrad,Nang,B)
    return hist_dict


def generate_sfcolumns(Nrad,Nang,B):
    col_name=[]
    for i in range(Nrad*B):
        name_rad='SFR'+str(i)
        col_name.append(name_rad)
    for j in range(2*Nang*B):
        name_ang='SFA'+str(j)
        col_name.append(name_ang)
    return col_name


def generate_X(G_rad,G_angle,Nrad,Nang,B,molecular):
    SF_col=generate_sfcolumns(Nrad,Nang,B)
    hist_dict=histogram_wACSFs(G_rad,G_angle,Nrad,Nang,B,molecular)
    SFresult=pd.DataFrame(hist_dict).T 
    SFresult.index=np.arange(len(molecular))
    SFresult.columns=SF_col
    return SFresult

if __name__ == "__main__": 
    Nrad,Nang,B=8,10,15        
    molecular_,data_,targets_,elements_=read_file("amidase_trial.txt")      
    G_rad=calc_rad_SF(Nrad,molecular_,data_,elements_)
    G_angle=calc_angular_mole_SFs(Nang,molecular_,data_,elements_)
    X=generate_X(G_rad,G_angle,Nrad,Nang,B,molecular_)
    X.to_csv('amidase_sf_trial.csv')
