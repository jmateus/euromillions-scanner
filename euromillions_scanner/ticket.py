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
        self.ticket_image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self._numbers = []
        self._stars = []

    def get_numbers(self):
        """
            Returns the array with the set of numbers of the ticket
        """
        return self._numbers

    def get_stars(self):
        """
            Returns the array with the set of lucky stars of the ticket
        """
        return self._stars
