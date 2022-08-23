from skimage.filters import threshold_otsu
from skimage.filters import sobel
from skimage.morphology import disk, remove_small_objects, binary_closing
from scipy.ndimage import binary_fill_holes

__all__ = ['segment_all_channels',
           'bright_field_segmentation',
           'fluorescent_segmentation']


def segment_all_channels(image, min_size=100, selem=disk(5)):
    """calculates the segmentation per channel
    
    It considers the first channel is brightfield and the rest are fluorescents

    Parameters
    ----------
    image : 3D array, shape (M, N,C)
        The input image with the first channel as brightfield and the rest as florescent 

    Returns
    -------
    segmented_image :  3D array, shape (M, N,C)
        The binary mask per channel

    Raises
    -------
    None
    """
    segmented_image = image.copy() * 0
    for ch in range(image.shape[2]):
        if ch == 0:
            segmented_image[:, :, ch] = bright_field_segmentation(image[:, :, ch],
                                                                  min_size=min_size,
                                                                  selem=selem)
        else:
            segmented_image[:, :, ch] = fluorescent_segmentation(image[:, :, ch],
                                                                 min_size=min_size,
                                                                 selem=selem)
    return segmented_image


def bright_field_segmentation(image, min_size=100, selem=disk(5)):
    """calculates the segmentation per channel using edge detection
    
    It first calculates the sobel filtered image to calculate the edges.
    Then removes the small objects, closes the binary shapes and 
    finally fills the shapes.

    Parameters
    ----------
    image : 2D array, shape (M, N)
        The input image only one channel 

    Returns
    -------
    segmented_image :  2D array, shape (M, N)
        The binary mask 

    Raises
    -------
    None

    References
    -------
    ..  [1] http://jkimmel.net/so-you-want-to-segment-a-cell/

    Notes
    -----
    1.  It works best for brightfield channels in Imaging Flow Cytometry (IFC)
    2.  We use otsu thresholding in the background

    """
    segmented_image = image.copy() * 0
    # calculate edges
    edges = sobel(image)

    # segmentation
    threshold_level = threshold_otsu(edges)
    bw = edges > threshold_level  # bw is a standard variable name for binary images

    # postprocessing
    bw_cleared = remove_small_objects(bw, min_size)  # clear objects <100 px

    # close the edges of the outline with morphological closing
    bw_close = binary_closing(bw_cleared, selem=selem)
    segmented_image = binary_fill_holes(bw_close)

    return segmented_image.astype(int)


def fluorescent_segmentation(image, min_size=100, selem=disk(5)):
    """calculates the segmentation using direct thresholding
    
    It calcualtes the threshold using otsu thresholding.
    Then removes the small objects, closes the binary shapes and 
    finally fills the shapes.

    Parameters
    ----------
    image : 2D array, shape (M, N)
        The input image with multiple channels. 

    Returns
    -------
    segmented_image :  2D array (int), shape (M, N) 
        Segmentation of the input image. 

    Raises
    -------
    None

    References
    -------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/applications/plot_human_mitosis.html

    Notes
    -----
    1.  It works best for florescent channels in Imaging Flow Cytometry (IFC).
    2.  We use otsu thresholding in the background

    """
    segmented_image = image.copy() * 0
    # segmentation
    threshold_level = threshold_otsu(image)
    bw = image > threshold_level  # bw is a standard variable name for binary images

    # postprocessing
    bw_cleared = remove_small_objects(bw, min_size)  # clear objects

    # close the edges of the outline with morphological closing
    bw_close = binary_closing(bw_cleared, selem=selem)
    segmented_image = binary_fill_holes(bw_close)

    return segmented_image.astype(int)
