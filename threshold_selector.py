import nibabel as nib
import napari
# %gui qt
import numpy as np
import argparse
import os
from dask import array as da


def split_filename(input_path):
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)

    base_arr = basename.split('.')
    ext = ''
    if len(base_arr) > 1:
        ext = base_arr[-1]
        if ext == 'gz':
            ext = '.'.join(base_arr[-2:])
        ext = '.' + ext
        basename = basename[:-len(ext)]
    return dirname, basename, ext


def threshold(image, t):
    arr = da.from_array(image, chunks=image.shape)
    return (arr > t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Threshold selector - Designed by Omar Al-Louzi')
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--images', type=str, nargs='+', required=True,
                        help='Path to reference anatomical image for threshold selection')
    required.add_argument('--membership', type=str, nargs='+', required=True,
                        help='Path to probability membership file')
    required.add_argument('--outdir', required=True,
                        help='Output directory where the resultant mask will be written')
    results = parser.parse_args()

    # Start by printing out some helpful info
    results.images = [os.path.expanduser(image) for image in results.images]
    dir, base, ext = split_filename(results.images[0])
    imgs = results.images

    lesion = [os.path.expanduser(image) for image in results.membership]

    '''
    Test commands: 
    results=['/home/allouzioa/Python_learning/Test/MID152_t01_20120621_FL_DS.nii.gz']; dir, base, ext = split_filename(results[0])
    lesion=['/home/allouzioa/Python_learning/Test/MID152_t01_20120621_T1_DS_CNNLesionMembership.nii.gz']
    imgs = results
    '''
    id = base[0:19]
    print("[INFO]: Now working on analyzing the following images:")
    for i in range(len(imgs)):
        imgs[i] = os.path.abspath(imgs[i])
        print(imgs[i])

    lesion = os.path.abspath(lesion[0])
    membership = nib.load(lesion).get_fdata().astype(np.float32)
    mem = membership
    # mem = np.multiply(membership, 255)

    # Now lets work on loading the images
    obj = nib.load(imgs[0]) # Load an example image to calculate dimensions
    origdim = np.asarray(obj.shape,dtype=int) # Extracts length of the np array shape
    vol = np.zeros(obj.shape + (len(imgs),), dtype=np.float32) # Create an empty container for files to be loaded.
    # Added brackets around len function and comma to make it a tuple

    for i, img in enumerate(imgs):
        temp = nib.load(img).get_fdata().astype(np.float32)
        vol[:,:,:,i] = temp
        temp=[]

    # Initialize napari with correct images
    with napari.gui_qt():
        # %gui qt
        # viewer = napari.view_image(vol[:, :, :, 2], name='T2star', order=(2, 1, 0))

        all_thresholds = da.stack([threshold(mem, t) for t in np.linspace(0, 1, num=100)])
        viewer = napari.view_image(vol[:, :, :, 0], name='Anat', order=(2, 1, 0), interpolation='lanczos')
        if len(imgs)>1:
            for i in range(1,len(imgs)):
                viewer.add_image(vol[:, :, :, i], name='Anat_'+str(i), interpolation='lanczos')
        viewer.add_image(all_thresholds,
                         name='thresholded', colormap='red', blending='additive', opacity=0.5
                         )

        thresh = input("Enter threshold: ")
        while (type(thresh) != float):
            try:
                thresh = float(thresh)
                if (thresh <0) | (thresh >1):
                    print("ERROR: Input entered is invalid: '{}'. Select a correct number between 0-1".format(thresh))
                    thresh = int(input("Enter lesion type: "))
            except ValueError:
                try:
                    thresh = int(thresh)
                    print("ERROR: Input entered is an integer: '{}'. Input should be an float".format(thresh))
                    thresh = input("Enter lesion type: ")
                except ValueError:
                    thresh = str(thresh)
                    print("ERROR: Input entered is a string: '{}'. Input should be an integer".format(thresh))
                    thresh = input("Enter lesion type: ")


    print("This is the end of the review. Please close the MRI viewer window.")


    print("Writing the output mask file...")
    outdir = os.path.abspath(os.path.expanduser(results.outdir))

    mask_outname = os.path.join(outdir, id + "_thresholded_binary_mask.nii.gz")
    mask = np.array(membership > thresh, dtype=int)
    nib.Nifti1Image(mask, obj.affine, obj.header).to_filename(mask_outname)

