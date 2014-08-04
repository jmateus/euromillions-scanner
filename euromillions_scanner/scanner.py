import cv2
import numpy as np


class Ticket:
    """
        Parses an image of a EuroMillions ticket and extracts the numbers and stars

        Attributes:
            ticket_image: OpenCV image of the ticket
            _numbers: Array with the sets of numbers in the ticket
            _stars: Array with the sets of lucky stars in the ticket
    """

    def __init__(self, image_path):
        """
            Loads the image located at image_path

            Args:
                image_path: a string containing the path to an image of a ticket
        """
        self._ticket_image = create_binary_image(image_path, (600, 600))
        self._numbers = []
        self._stars = []

    def get_numbers(self):
        """
            Returns the array with the sets of numbers of the ticket
        """
        return self._numbers

    def get_stars(self):
        """
            Returns the array with the sets of lucky stars of the ticket
        """
        return self._stars


"""General purpose auxiliary methods"""


def create_binary_image(image_path, max_size):
    """
        Opens the image located at image_path and creates a binary image from
        it, using Otsu's binarization.

        Args:
            image_path: a string containing the path to an image
            max_size: a tuple defining the maximum size of the image (width, height)

        Returns:
            A binary image
    """
    image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    binary_image = cv2.resize(image, max_size)  # TODO: Do not distort image
    (thresh, binary_image) = cv2.threshold(binary_image, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image


def invert_image(image):
    """
        Inverts a binary image

        Args:
            image: a binary image (black and white only)

        Returns:
            An inverted version of the image passed as argument
    """
    return 255 - image


def find_widest_contours(image, num_contours, dilation_scale=8):
    """
        Computes the widest contours of the image passed as parameter

        Args:
            image: the image we want to find the contours. the image should be
                binary, with black background and white foreground.
            num_contours: the number of contours we want to obtain
            dilation_scale: size of the kernel use to dilate the image
                horizontally. For example, if the dilation_scale is 8, the
                kernel is 8x8, with 3 middle rows with ones and the rest zeros

        Returns:
            The num_contours widest contours of the image
    """
    # TODO: try solving this using the Hough transform algorithm

    # Create the dilation matrix
    dilation_mat = np.zeros((dilation_scale, dilation_scale), np.uint8)
    dilation_mat[dilation_scale / 2 - 1:dilation_scale / 2 + 2] = 1

    # Create a copy because findContours modifies the image
    dilated_image = cv2.dilate(image, dilation_mat).copy()
    contours, _ = cv2.findContours(dilated_image,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    # Get the bounding rectangles of the countours
    contours_rects = map(cv2.boundingRect, contours)
    contours_rects.sort(key=lambda rect: rect[2])  # Sort by width
    contours_rects.reverse()  # Descending order

    return contours_rects[:num_contours]


def get_enclosed_rectangle(rect1, rect2):
    """
        Returns the rectangle enclosed between two non-overlapping rectangles

        Args:
            rect1: rectangle in the form of a tuple
                (top left x, top left y, width, height)
            rect2: same as rect1

        Returns:
            Enclosed rectangle in the same form as the parameters
    """
    min_x = min(rect1[0], rect2[0])
    min_y = min(rect1[1], rect2[1])

    width = max(rect1[0] - min_x + rect1[2], rect2[0] - min_x + rect2[2])
    # Height of the topmost rectangle
    top_height = rect1[3] if rect1[1] < rect2[1] else rect2[3]
    height = abs(rect1[1] - rect2[1]) - top_height

    return min_x, min_y + top_height, width, height


def show_image(image, wait_time=0, window_name='Image'):
    """
        Display an image using OpenCV

        Args:
            image: image we want to display
            wait_time: how long (in milliseconds) should the image be displayed
            window_name: name of the window (defaults to 'Image')
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)


def _main():
    img = create_binary_image('test/data/euro.jpg', (600, 600))
    inverted_img = invert_image(img)
    show_image(inverted_img, 2000)

    contours = find_widest_contours(inverted_img, 2)

    # Colored image used to draw the contours
    color_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    for rect in contours:
        point1 = (rect[0], rect[1])
        point2 = (rect[0] + rect[2], rect[1] + rect[3])

        cv2.rectangle(color_img, point1, point2, (0, 255, 0), 1)

    show_image(color_img)


if __name__ == '__main__':
    _main()