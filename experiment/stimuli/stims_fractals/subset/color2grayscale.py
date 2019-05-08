'''
Convert colored stimulus set (folder of images) into grayscale.

Creates a subdirectory called grayscale within the directory provided.

If images have alpha channels for transparency, add the --alpha flag.

Run with:
    python color2grayscale.py --ext jpg --dir /path/to/stimset
    or 
    python color2grayscale.py --ext jpg --dir /path/to/stimset --alpha
    if alpha channels present.
'''

import glob
import argparse

from skimage import io, color


parser = argparse.ArgumentParser(description='Create grayscale subdir of colored stim set.')
parser.add_argument('--ext', type=str, required=True, help='The file extension of all images (excluding period).')
parser.add_argument('--dir', type=str, required=True, help='The full path to stimulus set.')
parser.add_argument('--alpha', action='store_true', default=False, help='Maintain alpha channels of image, if present to start with.')
args = parser.parse_args()

IMPORT_DIR = args.dir
EXPORT_DIR = glob.os.path.join(IMPORT_DIR, 'grayscale')

EXTENSION = '.{:s}'.format(args.ext)

FILE_GLOB = glob.os.path.join(IMPORT_DIR, '*{:s}'.format(EXTENSION))
FILE_LIST = glob.glob(FILE_GLOB)

# create subdirectory
if glob.os.path.isdir(EXPORT_DIR):
    print '\nOVERRIDING EXISTING CROPPED SUBDIR!\n'
else:
    glob.os.mkdir(EXPORT_DIR)


# loop through each file, convert it, and save it out to new subdir

for f in FILE_LIST:

    color_img = io.imread(f)

    if args.alpha:
        grayscale_img = color.gray2rgb(color.rgb2gray(color_img), alpha=True)
        # mix the alpha channel back in
        grayscale_img[:,:,3] = color_img[:,:,3]/255. # need to scale alpha channel
    else:
        grayscale_img = color.rgb2gray(color_img)

    grayscale_fname = glob.os.path.join(EXPORT_DIR, glob.os.path.basename(f))

    io.imsave(grayscale_fname, grayscale_img)



# print out how many files were converted
N_FILES = len(FILE_LIST)
print '\nConverted {:d} images found in directory.\n'.format(N_FILES)

