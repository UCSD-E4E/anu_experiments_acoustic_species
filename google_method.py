# imports
import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import lap
from matplotlib.colors import Normalize

def get_2d(embeddings, identifier=None):
    # get 2d embeddings
    tsne = manifold.TSNE(random_state = 1, n_components=2, learning_rate=50)
    data2d = tsne.fit_transform(embeddings)

    data2d -= data2d.min(axis=0)
    data2d /= data2d.max(axis=0)
    
    plt.figure(figsize=(4, 4))
    plt.scatter(data2d[:,0], data2d[:,1], c=identifier, edgecolors='none', marker='o', s=12)  
    plt.show()
    
def plot_grid_transforms(embeddings, data):
    get_2d(embeddings)
    
    side = 10
    xv, yv = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    
    cost = cost * (10000000. / cost.max())
    
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=3) # Adjust vmin/vmax based on your data range

    min_cost, row_assigns, col_assigns = lap.lapjv(np.copy(cost))
    grid_jv = grid[col_assigns]
    print(col_assigns.shape)
    plt.figure(figsize=(4, 4))
    for index, (start, end) in enumerate(zip(data2d, grid_jv)):
        arrow_color = cmap(norm(data['site'][index]))
        
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                head_length=0.01, head_width=0.01, fc=arrow_color, ec=arrow_color)
    plt.show()
    
    

