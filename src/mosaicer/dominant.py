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

    def calculate(self, image: np.ndarray):
        clusters = KMeans(n_clusters=self.n_clusters)
        clusters.fit(image.reshape(-1,image.shape[-1]))

        # Get cluster proportion
        counter = Counter(clusters.labels_)
        
        # Get second most populous cluster 
        return clusters.cluster_centers_[sorted(counter.items())[1][0]]

    def calculate_array(self, images: Sequence[str]):
        images_dc = {}
        for image in images:
            im = Image.open(image)
            images_dc[image] = self.calculate(np.asarray(im))

        return images_dc

    def calculate_all(self, images_loc: str, colours_loc: str, labels_loc: str, processes: int, ):
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

    def get_nearest_label(self, colour: np.ndarray, no_repeat: bool = False):
        """Calculate minimum euclidian distance to 3D point"""
        distances = np.sqrt(np.sum((self.colours-colour)**2,axis=1))
        i = np.where(distances==np.amin(distances))[0][0]
        label = self.labels[i]
        if no_repeat:
            self.colours = np.delete(self.colours, i, axis=0)
            del self.labels[i]
        return label
