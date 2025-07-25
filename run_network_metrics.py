# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# %%
labels=['Left Precentral Gyrus',
 'RightPrecentral Gyrus',
 'Left Superior Frontal Gyrus',
 'Right Superior Frontal Gyrus',
 'Left Superior Orbital Gyrus',
 'Right Superior Orbital Gyrus',
 'Left Middle Frontal Gyrus',
 'Right Middle Frontal Gyrus',
 'Left Middle Orbital Gyrus',
 'Right Middle Orbital Gyrus',
 'Left Inferior Frontal Gyrus (p. Opercularis)',
 'Right Inferior Frontal Gyrus (p. Opercularis)',
 'Left Inferior Frontal Gyrus (p. Triangularis)',
 'Right Inferior Frontal Gyrus (p. Triangularis)',
 'Left Inferior Frontal Gyrus (p. Orbitalis)',
 'Right Inferior Frontal Gyrus (p. Orbitalis)',
 'Left Rolandic Operculum',
 'Right Rolandic Operculum',
 'Left SMA',
 'Right SMA',
 'Left Olfactory cortex',
 'Right Olfactory cortex',
 'Left Superior Medial Gyrus',
 'Right Superior Medial Gyrus',
 'Left Mid Orbital Gyrus',
 'Right Mid Orbital Gyrus',
 'Left Rectal Gyrus',
 'Right Rectal Gyrus',
 'Left Insula Lobe',
 'Right Insula Lobe',
 'Left Anterior Cingulate Cortex',
 'Right Anterior Cingulate Cortex',
 'Left Middle Cingulate Cortex',
 'Right Middle Cingulate Cortex',
 'Left Posterior Cingulate Cortex',
 'Right Posterior Cingulate Cortex',
 'Left Hippocampus',
 'Right Hippocampus',
 'Left ParaHippocampal Gyrus',
 'Right ParaHippocampal Gyrus',
 'Left Amygdala',
 'Right Amygdala',
 'Left Calcarine Gyrus',
 'Right Calcarine Gyrus',
 'Left Cuneus',
 'Right Cuneus',
 'Left Lingual Gyrus',
 'Right Lingual Gyrus',
 'Left Superior Occipital Gyrus',
 'Right Superior Occipital Gyrus',
 'Left Middle Occipital Gyrus',
 'Right Middle Occipital Gyrus',
 'Left Inferior Occipital Gyrus',
 'Right Inferior Occipital Gyrus',
 'Left Fusiform Gyrus',
 'Right Fusiform Gyrus',
 'Left Postcentral Gyrus',
 'Right Postcentral Gyrus',
 'Left Superior Parietal Lobule ',
 'Right Superior Parietal Lobule ',
 'Left Inferior Parietal Lobule ',
 'Right Inferior Parietal Lobule ',
 'Left SupraMarginal Gyrus',
 'Right SupraMarginal Gyrus',
 'Left Angular Gyrus',
 'Right Angular Gyrus',
 'Left Precuneus',
 'Right Precuneus',
 'Left Paracentral Lobule',
 'Right Paracentral Lobule',
 'Left Caudate Nucleus',
 'Right Caudate Nucleus',
 'Left Putamen',
 'Right Putamen',
 'Left Pallidum',
 'Right Pallidum',
 'Left Thalamus',
 'Right Thalamus',
 'Left Heschls Gyrus',
 'Right Heschls Gyrus',
 'Left Superior Temporal Gyrus',
 'Right Superior Temporal Gyrus',
 'Left Temporal Pole',
 'Right Temporal Pole',
 'Left Middle Temporal Gyrus',
 'Right Middle Temporal Gyrus',
 'Left Medial Temporal Pole',
 'Right Medial Temporal Pole',
 'Left Inferior Temporal Gyrus',
 'Right Inferior Temporal Gyrus',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Left Cerebellum',
 'Right Cerebellum',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis',
 'Cerebellar Vermis']

# %%
from nilearn.datasets import fetch_abide_pcp

abide_data=fetch_abide_pcp(
                            pipeline="cpac",
                            derivatives=["rois_ez"],
                            band_pass_filtering=True,
                            data_dir="path_to_store_abide_dataset"
                            )


all_signals=[abide_data["rois_ez"][i] for i in range(len(abide_data["rois_ez"]))]


# %%
#all functions related to metrics_list

def total_functional_wiring(graph):
    abs_weight_values=np.array([dic["abs_weight"] for dic in list(zip(*list(graph.edges(data=True))))[-1]])
    W=abs_weight_values.sum()

    return np.round(W,3)

def average_node_degree(graph):
    return np.round(np.array(list(dict(graph.degree(weight="abs_weight")).values())).mean(),4)

def argmax_node_degree(graph):
    return np.round(np.argmax(np.array(list(dict(graph.degree(weight="abs_weight")).values()))),3)


def weighted_global_cost_efficiency_ratio(graph):
    
    E_glob=0
    n=len(graph)
    for u in range(n):
        for v in range(u+1,n):

            E_glob+=(nx.shortest_path_length(graph,u,v,weight="cost"))**(-1)

    E_glob=2*E_glob/(n*(n-1))

    abs_weight_values=np.array([dic["abs_weight"] for dic in list(zip(*list(graph.edges(data=True))))[-1]])
    W=abs_weight_values.sum()

    return np.round(E_glob/W,4)

def weighted_smallworldness(graph):

    C=nx.average_clustering(graph)
    L=nx.average_shortest_path_length(graph)

    N=len(graph)
    E=len(graph.edges())

    p=2*E/(N*(N-1))

    random_graph=nx.erdos_renyi_graph(N,p)

    C_random=nx.average_clustering(random_graph)
    L_random=nx.average_shortest_path_length(random_graph)

    return np.round((C/L)/(C_random/L_random),4)

def max_node_betweeness(graph):
    ind_max=np.argmax(np.array(list(nx.betweenness_centrality(graph,weight="cost").values())))
    max_value=np.max(np.array(list(nx.betweenness_centrality(graph,weight="cost").values())))

    return np.round(max_value,4), ind_max



# %%
def all_metric_values(graph,rois_labels):
    metric_dict={}
    
    metric_dict["total functional wiring"]=total_functional_wiring(graph)
    metric_dict["characteristic length"]=np.round(nx.average_shortest_path_length(graph),3)
    metric_dict["average node degree"]=average_node_degree(graph)
    metric_dict["argmax node degree"]=rois_labels[argmax_node_degree(graph)]
    metric_dict["binarized average clustering"]=np.round(nx.average_clustering(graph),3)
    metric_dict["density"]=np.round(nx.density(graph),3)
    metric_dict["binarized small-worldness"]=weighted_smallworldness(graph)
    max_val, ind=max_node_betweeness(graph)
    metric_dict["max node betweenness"]=max_val
    metric_dict["argmax node betweenness"]=rois_labels[ind]
    
    return metric_dict

# %%
#handling all signals
from nilearn.connectome import ConnectivityMeasure
import networkx as nx

data_frame=[]

for i in range(len(all_signals)):
    signal=all_signals[i]
    connectome = ConnectivityMeasure(kind='correlation')
    adjacency_matrix = connectome.fit_transform([signal])[0]
    np.fill_diagonal(adjacency_matrix,0)
    adjacency_matrix[(adjacency_matrix>-0.2)&(adjacency_matrix<0.2)]=0
    subject_graph=nx.from_numpy_array(adjacency_matrix)
    
    if nx.is_connected(subject_graph):
        for u,v,data in subject_graph.edges(data=True):
            data["cost"]=1/(np.abs(data["weight"]))
            data["abs_weight"]=np.abs(data["weight"])

        metrics_dict=all_metric_values(subject_graph,labels)
        metrics_dict["label"]=abide_data["phenotypic"]["DX_GROUP"].iloc[i]
        data_frame.append(metrics_dict)
        
    del adjacency_matrix,subject_graph



df=pd.DataFrame(data_frame)

df.to_csv("network_metrics.csv")


