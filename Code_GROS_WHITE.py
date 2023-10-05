import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pickle
from matplotlib import style
from numpy.core.numeric import Inf
from numpy.testing._private.utils import tempdir

#les données que nous avons utilisé étaient trop volumineuses à envoyer dans le dossier. On peut les trouver à l'adresse:
#https://github.com/yhu01/PT-MAP et nous avons utilisé le jeu de données minilmageNet
path_file = r"C:\Users\Jack\Desktop\CODEV\PT-MAP-master\output.plk" #Penser à changer le chemin d'accès

with open(path_file,'rb') as f:
    data = pickle.load(f)

def predict(classes, features,filtered):
    centroids = []
    classifications = [20]*nb_features
    features_copy = list(features)
    features_norm = np.linalg.norm(features_copy, axis=0)

    for i, c in enumerate(classes):       
        centroids.append(np.average(c, axis = 0))

    distances = np.zeros((nb_features,nb_classes))
    deja_vu = np.zeros((nb_classes,nb_features),dtype=bool)

    #initialisation d'une matrice des distances entre les features et les centroïdes
    for i, centroid in enumerate(centroids):
        for j, feature in enumerate(features_copy):
            distances[j,i] = np.linalg.norm(feature - centroid)

    temp_centroids=classes
    while false(features_copy):  # = tant que la liste des features ne contient pas que des éléments vides
        
        for jind,feature in enumerate(features_copy):
           
            if len(feature)==0:
                pass

            else:
                #récupération de l'indice du centroïde le plus proche de chaque feature
                closest_centroid_distance = np.min(distances[jind])                    
                closest_centroid_index = int(np.where(distances[jind] == closest_centroid_distance)[0])
                
                #si le feature avait déjà été affecté temporairement à la classe, on l'ajoute définitivement
                if  deja_vu[closest_centroid_index,jind] :
                    classifications[jind] = closest_centroid_index
                    

                    #filtrage dynamique 
                    if filtered:
                        c = classes[closest_centroid_index]
                        c.append(features_copy[jind])
                        cluster_filtered = filter(c)
                        centroids[closest_centroid_index] = np.average(cluster_filtered, axis = 0)

                    #pour ne plus prendre en compte le feature qui est devenu labellisé, on le remplace par une liste vide
                    features_copy[jind]=[]

                #si le feature n'a jamais encore été détecté comme le plus proche de la classe, on l'affecte temporairement
                #en s'en servant uniquement pour mettre à jour le centroïde correspondant       
                else: 
                    deja_vu[closest_centroid_index,jind] = True
                    temp_centroids[closest_centroid_index].append(feature)

        #mise à jour des centroïdes
        for i, centroid in enumerate(centroids):
            centroids[i]=np.average(temp_centroids[i],axis=0)

            #mise à jour de la matrice des distances
            for j, feature in enumerate(features_copy):
                if len(feature)==0:
                       distances[j,i] = Inf
                else :
                    distances[j,i] = np.linalg.norm(feature - centroid)     

    return classifications

def false(a):
    for _ in range(len(a)):
        if len(a[_])!=0:
            
            return True
    return False

def s(x,y):
    return np.exp(-np.linalg.norm(x - y)**2/10)

def similarity(F):
    n = len(F)
    S = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            S[i,j] = s(F[i],F[j]) #calcul de la similarité entre chaque feature

    somme = np.sum(S)
    S /= somme #division par la norme L1  

    return S


def W_new(S,k=5):
    n = np.shape(S)[0]
    W = np.zeros((n,n))
    
    #On sélectionne les k plus grands facteurs d'une ligne
    for i in range(n):
        L_sort = sorted(list(S[i,:]))[::-1] #liste décroissante des facteurs d'une ligne
        L_new = []

        #on prend les k plus grand facteurs
        for m in range(k):
            L_new.append(L_sort[m]) 

        for j in range(n):
            if S[i,j] in L_new:
                W[i,j],W[j,i] = S[i,j],S[j,i] #par symétrie 
    return W #W est bien symétrique


def filtrage_graphe(F,W): 
    n = np.shape(W)[0]
    D = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            D[i,i] += W[i,j] #matrice de degré
        D[i,i] = 1/np.sqrt(D[i,i]) #D^-(1/2)

    L = np.eye(n) - np.linalg.multi_dot((D,W,D)) #laplacien normalisé
    U = np.linalg.eig(L)[1] #matrice de passage orthogonale
    U_trans = np.transpose(U) 



    H = np.zeros((n,n))
    vp = np.linalg.eig(L)[0] #vecteur des valeurs propres du laplacien normalisé
    vp_moy=max(vp)-(max(vp)-np.average(vp))/4
        
    for i in range(n):   
        H[i,i]=np.exp(-vp[i]*1.2)

    return np.linalg.multi_dot((U,H,U_trans,F)) #features filtrés

def filter(c):
    S = similarity(c)
    W = W_new(S)
    c_filtered = filtrage_graphe(c,S) 
    return c_filtered


taille_feature = data[80][0]
nb_test = 2000
nb_classes = 5
nb_labelled = 5 
nb_non_labelled = 15
nb_features = nb_classes*nb_non_labelled 
effi=[]
list_classes = [0]*nb_non_labelled+[1]*nb_non_labelled+[2]*nb_non_labelled+[3]*nb_non_labelled+[4]*nb_non_labelled

for i in range(nb_test):
    labelled_data = []
    non_labelled_data = []
    classes = []

    #tirage au sort des classes et on évite les doublons
    #rd.seed(0) les seeds permettent de retravailler sur même ensemble (pas d'aleatoire) pour reproduire des résultats
    random_classes = rd.sample(range(80,99),nb_classes)
    classes_all_data=[]
    #affectation des DL et des DNL
    for rand in random_classes:
        #rd.seed(0)
        data_class = rd.sample(data[rand], nb_labelled + nb_non_labelled)
        classes_all_data+=data_class[0 : nb_labelled]
        non_labelled_data+=data_class[nb_labelled : nb_labelled + nb_non_labelled]

    




    classes_all_data=list(filter(classes_all_data)) #filtrage de toutes classes ensembles

    
    for i in range(nb_classes):   
        classes.append(classes_all_data[i*nb_labelled:i*nb_labelled+nb_labelled]) #redéfinit les classes filtrés

    result = predict(classes,non_labelled_data,False) #lance predict, mettre True pour le filtrage dynamique mais très demandant pour l'ordinateur
    
    

    nf_efficiency_k = 0
    nf_verification = np.array(list_classes) - result
    for nf_diff in nf_verification:
            if nf_diff == 0:
                nf_efficiency_k += 1
    effi.append(nf_efficiency_k/75*100)
    print(nf_efficiency_k)
    
print(np.average(effi),effi)
plt.grid()
plt.hist(effi,color='#0504aa',alpha=0.7, rwidth=0.5)

plt.show()