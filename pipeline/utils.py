import librosa, os
from scipy.linalg import toeplitz
from scipy.stats import zscore
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist

from statsmodels.tsa.stattools import acf

from pyha_analyzer import extractors
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessorsNew

import lap

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from tqdm import tqdm

import torchvision.transforms as transforms


# juan colonna entropy

def Entropy(p1):
    p1 = p1/np.sum(p1)
    return entropy(p1)/np.log(len(p1))
# EGCI calculation from https://github.com/juancolonna/EGCI/blob/master/Example_of_EGCI_calculation.ipynb

def JSD(p):
    n = len(p)
    q = np.ones(n)/n # Uniform reference
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/p.sum() # normalize
    m = (p + q) / 2
    jensen0 = -2*((((n+1)/n)*np.log(n+1)-2*np.log(2*n) + np.log(n))**(-1))
    return jensen0*(entropy(p, m) + entropy(q, m)) / 2

def EGCI(x, lag):
    x = zscore(x)
    
    # Algorithm steps 
    rxx = acf(x, nlags=lag, adjusted=True, fft=True) #https://github.com/blue-yonder/tsfresh/issues/902
    Sxx = toeplitz(rxx)
    U, s, Vt = np.linalg.svd(Sxx) #svd(Sxx)
    
    return Entropy(s), Entropy(s)*JSD(s) 

def get_egci(row, split=False):
    path = row['filepath']
    label = 'degraded' if str(row['labels']) == '[1, 0]' else 'non_degraded'
    dataset = row['dataset']
    
    if (".WAV" not in path) and (".wav" not in path):
        print("directory found")
        return None
    
    output_list = []
    
    audios = get_audio_split(path, 5)
    
    if audios == []:
        print("no audio here :(")
        return None
    
    if split:
        for audio in audios:
            h, c = EGCI(audio)

            output_data = {
                    "path": path,
                    "offset_s": 0,
                    "gt": label,
                    "site": dataset,
                    "entropy": h,
                    "complexity": c
                }
            output_list.append(output_data)
        
    else:
        h, c = EGCI(audios[0])

        output_data = {
                "path": path,
                "offset_s": 0,
                "gt": label,
                "site": dataset,
                "entropy": h,
                "complexity": c
            }
        output_list.append(output_data)
        
        
        
    return output_list

def extract_coral(path, duration, preprocess=False):
    coralreef_extractor = extractors.MultiCoralReef()
    coral_ads = coralreef_extractor(path)
    preprocessor = MelSpectrogramPreprocessorsNew(duration=duration, class_list=coral_ads["train"].features["labels"].feature.names)

    if preprocess:
        coral_ads["train"].set_transform(preprocessor)
        coral_ads["valid"].set_transform(preprocessor)
        coral_ads["test"].set_transform(preprocessor)
    
    return (coral_ads['train'], coral_ads['valid'], coral_ads['test'])
    
def get_audio_split(filepath, duration):
    audio, sr = librosa.load(filepath, sr=None)
    
    block_size = sr * duration
    
    blocks = []
    
    for block in range(len(audio) // block_size):
        blocks.append(audio[(block_size * block):(block_size * block) + block_size])
    
    return blocks


def get_melspec(audio, sample_rate):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate,
        n_fft=2048, 
        hop_length=256, 
        power=2.0, 
        n_mels=256,
    )

    pillow_transforms = transforms.ToPILImage()
    mel_image = np.array(pillow_transforms(mel), dtype=np.float32)[np.newaxis, ::] / 255
    
    return mel_image
    





class spectrogram_grid():
    """
    Plot melspectrograms of a set of embeddings
    
    Note: since linear optimization requires a square matrix, we can only go up by squares
    
    Parts of implementation from Google
    """
    
    def __init__(self, data, paths, labels=None, normalize=False):
        self.data_2d=data
        self.labels=labels
    
    
        if normalize:
            data2d -= data2d.min(axis=0)
            data2d /= data2d.max(axis=0)
        
        
        plt.figure(figsize=(4, 4))
        plt.scatter(data2d[:,0], data2d[:,1], edgecolors='none', marker='o', s=12)
        plt.show()

        side = 10
        xv, yv = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        grid = np.dstack((xv, yv)).reshape(-1, 2)
        
        
        cost = cdist(grid, data2d, 'sqeuclidean')
        cost = cost * (10000000. / cost.max())

        _, _, col_assigns = lap.lapjv(np.copy(cost))
        self.grid_jv = grid[col_assigns]

        self.coords = (self.grid_jv*9).astype(int)
        
        self.cmap = cm.viridis
        self.norm = Normalize(vmin=0, vmax=3) # Adjust vmin/vmax based on your data range
        
    def plot_arrows(self):
        plt.figure(figsize=(4, 4))
        for index, (start, end) in enumerate(zip(self.data2d, self.grid_jv)):
            arrow_color = self.cmap(1)
            
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_length=0.01, head_width=0.01, fc=arrow_color, ec=arrow_color)
        plt.show()
        
    def plot_spectrogram(self):
        fig = plt.figure(figsize=(16., 16.))
        
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),
                 axes_pad=0,  # pad between Axes in inch.
                 )

        indexes = np.lexsort((self.coords[:, 0], -self.coords[:, 1]))

        print(self.coords)

        images = np.array(self.paths)[indexes]

        for ax, item in tqdm(zip(grid, images)):
            # spectrogram_tensor = torch.tensor(item, dtype=torch.float32)
            # im = spectrogram_tensor.squeeze(0).numpy()  # shape: (H, W)
            
            
            y, sr = librosa.load(item, sr = None)
            im = librosa.feature.melspectrogram(y=y, sr=sr)

            im = librosa.power_to_db(im, ref=np.max)
            
            im = im[:,0:256]
            
            # print(im)  # Optionally print for debugging
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
        plt.show()
    
    
    
    
    