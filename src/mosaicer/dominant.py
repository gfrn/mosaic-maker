from __future__ import print_function
import json
import multiprocessing
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from typing import Counter, List, Sequence
from sklearn.cluster import KMeans
from itertools import batched

class DominantColourCalculator:
    def __init__(self, n_clusters: int = 4, ):
        self.n_clusters = n_clusters

    @property
    def colours(self) -> np.ndarray:
        return self._colours
    
    @colours.setter
    def colours(self, values: np.ndarray):
        self._colours = values

    @property
    def labels(self) -> List[str]:
        return self._labels
    
    @labels.setter
    def labels(self, values: List[str]):
        self._labels = values

    def calculate(self, image: np.ndarray, cluster_index: int = 1):
        """Calculate n-th most populous colour cluster
        
        Args:
            image: Image data to parse, as Numpy array
            cluster_index: Cluster to get average colour from. Starts from 0 (most populous)
            
        Returns:
            Average colour of selected cluster (as 1, 3 Numpy array)"""
        clusters = KMeans(n_clusters=self.n_clusters)
        clusters.fit(image.reshape(-1,image.shape[-1]))

        # Get cluster proportion
        counter = Counter(clusters.labels_)
        
        # Get second most populous cluster 
        return clusters.cluster_centers_[sorted(counter.items())[cluster_index][0]]

    def calculate_array(self, images: Sequence[str]) -> dict[str, np.ndarray]:
        """Load data and calculate dominant colour for all images in array
        
        Args:
            images: Paths to images to process
        
        Returns:
            dict object with image paths as keys, and the dominant colour as values."""
        images_dc = {}
        for image in images:
            im = Image.open(image)
            images_dc[image] = self.calculate(np.asarray(im))

        return images_dc

    def calculate_all(self, images_loc: str, colours_loc: str, labels_loc: str, processes: int, ):
        """Calculate all colours in a directory, and save labels/colour values to files.
        
        Args:
            images_loc: Location of folder with images to parse
            colours_loc: File path for new colour data file
            labels_loc: File path for new labels data file
            processes: Number of processes to use for parallel processing"""
        images = [join(images_loc, f) for f in listdir(images_loc) if isfile(join(images_loc, f))]
        images_b = batched(images, 25)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(self.calculate_array, images_b)
        
        dict_results = {}
        for r in results:
            dict_results.update(r)
        vals = np.array(list(dict_results.values()))
        np.save(colours_loc, vals)
        with open(labels_loc, 'w') as f:
            json.dump(list(dict_results.keys()), f)

    def get_nearest_label(self, colour: np.ndarray, no_repeat: bool = False) -> str:
        """Get image whose dominant colour most closely matches the passed colour. 
        This is done by calculating the minimum euclidian distance to each colour in 'colours'.
        
        Args:
            colour: Colour to find
            no_repeat: Do not reuse the matched image in the future
            
        Returns:
            Path to image"""
        distances = np.sqrt(np.sum((self.colours-colour)**2,axis=1))
        i = np.where(distances==np.amin(distances))[0][0]
        label = self.labels[i]
        if no_repeat:
            self.colours = np.delete(self.colours, i, axis=0)
            del self.labels[i]
        return label
