import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection
import matplotlib.cm as cm


def construct_u_matrix(weights, hex_side):
    """
    Convert weights matrix to U-matrix
    The U-matrix has dimensions lattice_dim*2-1
    Args:
        weights (np.array): dimensions lattice_dim[0] * lattice_dim[1] * features_dim
        hex_side (float): length of one side of a hexagon in the U-matrix
    Returns:
        hexagon_offsets (np.array): x,y coordinats of hexagons in the U-matrix, 
                                    dimensions ???
        hexagon_color_matrix (np.array): color of each hexagon in U-matrix
                                         dimensions lattice_dim*2-1
    """

    hex_width, hex_height, hex_area = hexagon_dims(hex_side)

    umatrix_dim = np.asarray(lattice_dim)*2-1

    # x,y coordinates of hexagons in U-matrix
    hexagon_offsets = []
    # color of each hexagon in U-matrix
    hexagon_color_matrix = []
    indices_check = np.zeros(umatrix_dim, dtype=object)

    for i in range(umatrix_dim[0]):
        for j in range(umatrix_dim[1]):
            if (i%2==0) and (j%2==0):
                i1,i2 = get_hexagonal_indices(i)
                j1,j2 = get_hexagonal_indices(j)
                hexagon_color_matrix.append(n_samples_per_node[i1][j1])
                indices_check[i,j] = '{},{}'.format(i1,j1)
            else:
                i1,i2 = get_hexagonal_indices(i)
                j1,j2 = get_hexagonal_indices(j)
                hexagon_color_matrix.append(np.linalg.norm(weights[i1,j1]-weights[i2,j2]))
                indices_check[i,j] = '{},{}-{},{}'.format(i1,j1,i2,j2)
            x_hex = i * hex_width + 0.5 * hex_width * j
            y_hex = j * hex_height    
            hexagon_offsets.append([x_hex,y_hex])

    hexagon_offsets = np.asarray(hexagon_offsets)
    hexagon_color_matrix = np.asarray(hexagon_color_matrix)
    
    return hexagon_offsets, hexagon_color_matrix


def plot_u_matrix(hexagon_offsets, hexagon_color_matrix, hex_side):
    dpi = 72.0
    width = 30
    x, y = umatrix_dim
    xinch = x * width / dpi
    yinch = y * width / dpi
    fig = plt.figure(figsize=(xinch, yinch), dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')

    _, _, hex_area = hexagon_dims(hex_side)
    collection_bg = RegularPolyCollection(
        numsides=6,  # a hexagon
        rotation=0,
        sizes=(hex_area,),
        edgecolors = None,
        array = hexagon_color_matrix,
        cmap = cm.Greys,
        offsets = hexagon_offsets,
        transOffset = ax.transData,
    )
    ax.add_collection(collection_bg, autolim=True)

    ax.axis('off')
    ax.autoscale_view()
    plt.colorbar(collection_bg)
    
    return ax


# https://www.redblobgames.com/grids/hexagons/

def hexagon_area(side):
    area = 3*np.sqrt(3)/2*side**2
    return area


def hexagon_dims(side):
    
    # side = radius in a hexagon
    width = np.sqrt(3) * side
    height = 2 * side
    area = 3*np.sqrt(3)/2*side**2
    
    return width, height, area


def get_hexagonal_indices(i):
    """
    Map from lattice node indices to u-matrix indices
    """
    if (i%2==0):
        i1 = i/2
        i2 = i/2
    else:
        i1 = np.floor(i/2)
        i2 = np.ceil(i/2)

    return int(i1), int(i2)


def plot_profile_per_node(average_vector_per_node, samples_per_node, n_samples_per_node, X, weights):
    lattice_dim = n_samples_per_node.shape
    features_dim = weights.shape[2]

    threshold = 1
    iplot = 1
    for i in range(lattice_dim[0]):
        for j in range(lattice_dim[1]):
            if n_samples_per_node[i,j] > threshold:
                if n_samples_per_node[i,j]> 100000:
                    continue
                plt.subplot(100,1,iplot)
                iplot += 1
                print(n_samples_per_node[i][j])
                print(samples_per_node[i][j])
                print(i,j)
                for k in samples_per_node[i][j]:
                    # plot all individual samples
                    plt.plot(range(features_dim), X[k], color='grey', alpha=0.4)
                plt.plot(range(features_dim), average_vector_per_node[i,j], label='average')
                plt.plot(range(features_dim), weights[i,j], label='weights')
                plt.legend()
    
    fig = plt.gcf()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1]*iplot*5)
    plt.show()
