import cv2


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


def show_image(image, wait_time=0, window_name='Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)


if __name__ == '__main__':
    img = create_binary_image('test/data/euro2.jpg', (600, 600))
    show_image(img)