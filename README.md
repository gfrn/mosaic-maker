[![PyPI](https://img.shields.io/pypi/v/mosaicer.svg)](https://pypi.org/project/mosaicer)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# mosaicer

Tool to make mosaics from a folder of images

Source          | <https://github.com/gfrn/mosaicer>
:---:           | :---:
PyPI            | `pip install mosaicer`
Releases        | <https://github.com/gfrn/mosaicer/releases>


## Usage (command line)

<img src="https://github.com/gfrn/mosaic-maker/docs/pepper.png" width=300 height=300/>

*Picture of bell pepper formed by protein structures*

### Installation

`pip install .`

### Options

* --clusters: Number of clusters to use for KMeans clustering for dominant colour detection
* --processes: Number of processes to use for multithreaded processing
* --images: Location of images folder
* --labels: Location of labels file (.json)
* --colours: Location of colours file (.npy)
* --piece_size: Mosaic piece size (in pixels)
* --resize_ratio: Ratio between the number of images used to form the larger image, and the original pixel count
* --out_file: Output image
* --in_file: Input image
* --no_repeat: Do not reuse images to form mosaic

## Usage (library)

Check the `DominantColourCalculator` class attribute docs for more information

```py
from mosaicer.dominant import DominantColourCalculator
dcc = DominantColourCalculator(n_clusters=4)

im = Image.open(IMAGE_PATH_HERE)
dominant_colour = dcc.calculate(np.asarray(im))
```
