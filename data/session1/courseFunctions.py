import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import csv

colours = ['k', 'r', 'b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class plots:
    def __init__(self, PATH, FILES):
        self.smiles, self.data_colours, self.labels = setup_data(PATH, FILES)
        self.pca = _calc_PCA(self.smiles)
        self.umap = _calc_UMAP(self.smiles)

    def plot_pca(self):
        plot_map(np.transpose(self.pca)[0], np.transpose(self.pca)[1], self.data_colours, self.labels)

    def plot_umap(self):
        plot_map(np.transpose(self.umap)[0], np.transpose(self.umap)[1], self.data_colours, self.labels)

def setup_data(PATH, FILES):
    dfs_with_colours = _set_colours(PATH, FILES)
    smiles, data_colours = _standardize_smiles(dfs_with_colours)
    labels = _get_labels(FILES)
    return smiles, data_colours, labels

def _set_colours(PATH, FILES):
    smiles_dfs = _get_dfs(PATH, FILES)

    for i, df in enumerate(smiles_dfs):
        if i < len(colours):
            df['colours'] = colours[i]
        else:
            df['colours'] = 'k'

    return smiles_dfs

def _get_dfs(PATH, FILES):
    dataframes = []
    files_list = FILES
    if type(FILES) != list:
        files_list = [FILES]

    for f in files_list:
        file_path = os.path.join(PATH, f)
        dataframes.append(pd.read_csv(file_path, sep=_find_separator(file_path)))
    smiles_df_list = [df[["Smiles"]] for df in dataframes]
    return smiles_df_list

def _find_separator(file_path):
    with open(file_path, 'r') as csvfile:
        try:
            delimiter = csv.Sniffer().sniff(csvfile.read(1024), delimiters=[',', ';']).delimiter
            return delimiter
        except:
            return ','

def _standardize_smiles(df_list):
    combined_df = pd.concat(df_list)
    combined_df.reset_index(level=None, drop=True, inplace=True)

    X_combined = []
    colours_combined = combined_df.colours.values.tolist()

    for i, s in enumerate(combined_df["Smiles"]):
        try:
            mol = Chem.MolFromSmiles(s)
            X_combined.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))
        except:
            colours_combined.pop(i)

    return X_combined, colours_combined

def _get_labels(FILES):
    labels_list = [f[:-4] for f in FILES]
    return labels_list

def _calc_PCA(smiles):
    pca = PCA(n_components=2)
    return pca.fit_transform(smiles)

def _calc_UMAP(smiles):
    umap_transformer = umap.UMAP(n_neighbors=15, min_dist=0.8)
    return umap_transformer.fit_transform(smiles)

def plot_map(X, y, data_colours, labels):
    plt.clf()
    fig = plt.figure()

    plt.scatter(X, y, color = data_colours, s = 5)
    #plt.title (title,fontsize=14,fontweight='bold',family='sans-serif')
    plt.xlabel ("Dimension 1",fontsize=14,fontweight='bold')
    plt.ylabel ("Dimension 2",fontsize=14,fontweight='bold')

    legend_elements = []
    for i, l in enumerate(labels):
        legend_elements.append(Line2D([0], [0], marker='.', color='w', label=l, markerfacecolor=colours[i], markersize=10))
    plt.legend(handles=legend_elements, frameon=False)

    plt.tick_params ('both',width=2,labelsize=12)
    plt.tight_layout()
    plt.show()

