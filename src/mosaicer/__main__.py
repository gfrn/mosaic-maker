from __future__ import print_function
import json
from PIL import Image
import numpy as np
from os.path import isfile
import argparse
from .dominant import DominantColourCalculator

arg_parser = argparse.ArgumentParser(description='Generate a mosaic from a list of images')

arg_parser.add_argument("--clusters", type=int, help="Number of clusters to use for KMeans clustering for dominant colour detection", nargs='?', default=4)
arg_parser.add_argument("--processes", type=int, help="Number of processes to use for multithreaded processing", nargs='?', default=4)
arg_parser.add_argument("--images", type=str, help="Location of images folder", nargs='?', default="images")
arg_parser.add_argument("--labels", type=str, help="Location of labels file (.json)", nargs='?', default="labels.json")
arg_parser.add_argument("--colours", type=str, help="Location of colours file (.npy)", nargs='?', default="colours.npy")
arg_parser.add_argument("--piece_size", type=int, help="Mosaic piece size (in pixels)", nargs='?', default=32)
arg_parser.add_argument("--resize_ratio", type=int, help="Ratio between the number of images used to form the larger image, and the original pixel count", nargs='?', default=10)
arg_parser.add_argument("--out_file", type=str, help="Output image", nargs='?', default="out.png")
arg_parser.add_argument("--in_file", type=str, help="Input image", nargs='?', default="in.png")
arg_parser.add_argument("--no_repeat", type=bool, help="Do not reuse images to form mosaic", nargs='?', default=False)

pargs = arg_parser.parse_args()

if __name__ == "__main__":
    dcc = DominantColourCalculator(n_clusters=pargs.clusters)
    if not isfile(pargs.labels) or not isfile(pargs.colours):
        print("Label/colour files not found, calculating...")
        dcc.calculate_all(pargs.images, pargs.colours, pargs.labels, pargs.processes)

    print("Loading labels/colour information...")
    colours = np.load(pargs.colours)
    dcc.colours = colours
    with open(pargs.labels, "r") as f:
        labels = json.load(f)
        dcc.labels = labels

    print(f"Resizing {pargs.in_file}")
    im = Image.open(pargs.in_file)
    im = im.resize((im.width//pargs.resize_ratio,im.height//pargs.resize_ratio))
    base_arr = np.asarray(im)

    print("Getting closest pixel colour matches...")
    results = np.apply_along_axis(dcc.get_nearest_label, 1, base_arr.reshape(-1, 3))
    dst = Image.new('RGB', (im.width*pargs.piece_size, im.height*pargs.piece_size))
    
    print("Creating output image...")
    for i, image in enumerate(results):
        piece = Image.open(str(image))
        piece = piece.resize((pargs.piece_size, pargs.piece_size))
        dst.paste(piece, (i%im.width*pargs.piece_size, i//im.width*pargs.piece_size))

    dst.save(pargs.out_file)
    print(f"Saved to {pargs.out_file}!")
    