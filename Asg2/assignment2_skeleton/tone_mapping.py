#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 2
# Name :
# Student ID :
# Email Addr :
#

import cv2
import numpy as np
import os
import sys
import argparse


class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message

    """

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def hdr_read(filename: str) -> np.ndarray:
    """ Load a hdr image from a given path

    :param filename: path to hdr image
    :return: data: hdr image, ndarray type
    """
    data = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    assert data is not None, "File {0} not exist".format(filename)
    assert len(data.shape) == 3 and data.shape[2] == 3, "Input should be a 3-channel color hdr image"
    return data


def ldr_write(filename: str, data: np.ndarray) -> None:
    """ Store a ldr image to the given path

    :param filename: target path
    :param data: ldr image, ndarray type
    :return: status: if True, success; else, fail
    """
    return cv2.imwrite(filename, data)


def compute_luminance(input: np.ndarray) -> np.ndarray:
    """ compute the luminance of a color image

    :param input: color image
    :return: luminance: luminance intensity
    """
    luminance = 0.2126 * input[:, :, 0] + 0.7152 * input[:, :, 1] + 0.0722 * input[:, :, 2]
    return luminance


def map_luminance(input: np.ndarray, luminance: np.ndarray, new_luminance: np.ndarray) -> np.ndarray:
    """ use contrast reduced luminace to recompose color image

    :param input: hdr image
    :param luminance: original luminance
    :param new_luminance: contrast reduced luminance
    :return: output: ldr image
    """
    # write you code here
    # to be completed
    luminance = np.maximum(luminance, 1e-4)

    ratio = new_luminance / luminance

    output = input * ratio[:, :, np.newaxis]

    output = np.clip(output, 0, 1)
    # write you code here
    return output


def he_tonemap(input: np.ndarray) -> np.ndarray:
    """ global tone mapping with histogram equalization operator

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """
    # write you code here
    # to be completed
    original_luminance = compute_luminance(input)

    # Compute the histogram
    hist, bins = np.histogram(original_luminance.flatten(), bins=256, range=[np.min(original_luminance), np.max(original_luminance)], density=True)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize the cumulative distribution function to [0, 1]

    # linear interpolation
    new_luminance = np.interp(original_luminance.flatten(), bins[:-1], cdf_normalized)
    new_luminance = new_luminance.reshape(original_luminance.shape)

    output = map_luminance(input, original_luminance, new_luminance)
    # write you code here
    return output


def bilateral_filter(input: np.ndarray, size: int, sigma_space: float, sigma_range: float) -> np.ndarray:
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: input image/map
    :param size: windows size for spatial filtering
    :param sigma_space: filter sigma for spatial kernel
    :param sigma_range: filter sigma for range kernel
    :return: output: filtered output
    """
    # write you code here
    # to be completed
    # If d is non-positive, it is computed from 'sigmaSpace'
    output = cv2.bilateralFilter(input, d=-1, sigmaColor=sigma_range, sigmaSpace=sigma_space)
    # write you code here
    return output


def durand_tonemap(input: np.ndarray) -> np.ndarray:
    """ local tone mapping with durand's operator (bilateral filtering)

    :param input: hdr image
    :return: output: ldr image, value range [0, 1]
    """
    # write you code here
    # to be completed

    sigma_space = 0.02 * min(input.shape[:2])
    sigma_range = 0.4  # As specified
    contrast = 50

    L = compute_luminance(input)

    log_L = np.log10(L + 1e-4)  # Adding a small value to avoid log(0)

    base_layer = bilateral_filter(log_L.astype(np.float32), size=5, sigma_space=sigma_space, sigma_range=sigma_range)

    detail_layer = log_L - base_layer

    gamma = np.log10(contrast) / (np.max(base_layer) - np.min(base_layer))

    D_prime = 10 ** (gamma * base_layer + detail_layer)

    D = D_prime * 1 / 10 ** np.max(gamma * base_layer)

    output = map_luminance(input, L, D)

    # write you code here
    return output


# operator dictionary
op_dict = {
    "durand": durand_tonemap,
    "he": he_tonemap
}

if __name__ == "__main__":
    # read arguments
    parser = ArgParser(description='Tone Mapping')
    parser.add_argument("filename", metavar="HDRImage", type=str, help="path to the hdr image")
    parser.add_argument("--op", type=str, default="all", choices=["durand", "he", "all"],
                        help="tone mapping operators")
    args = parser.parse_args()
    # print banner
    banner = "CSCI3290, Spring 2024, Assignment 2: tone mapping"
    bar = "=" * len(banner)
    print("\n".join([bar, banner, bar]))
    # read hdr image
    image = hdr_read(args.filename)


    # define the whole process for tone mapping
    def process(op: str) -> None:
        """ perform tone mapping with the given operator

        :param op: the name of specific operator
        :return: None
        """
        operator = op_dict[op]
        # tone mapping
        result = operator(image)
        # gamma correction
        result = np.power(result, 1.0 / 2.2)
        # convert each channel to 8bit unsigned integer
        result_8bit = np.clip(result * 255, 0, 255).astype('uint8')
        # store the result
        target = "output/{filename}.{op}.png".format(filename=os.path.basename(args.filename), op=op)
        msg_success = lambda: print("Converted '{filename}' to '{target}' with {op} operator.".format(
            filename=args.filename, target=target, op=op
        ))
        msg_fail = lambda: print("Failed to write {0}".format(target))
        msg_success() if ldr_write(target, result_8bit) else msg_fail()


    if args.op == "all":
        [process(op) for op in op_dict.keys()]
    else:
        process(args.op)
