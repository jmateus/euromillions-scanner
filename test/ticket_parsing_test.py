import unittest
import nose
import os
import json
import textwrap
import euromillions_scanner as euro


class TicketParsingTest(unittest.TestCase):
    """
        Uses the sample data to test the code.

        Attributes:
            data_json: JSON with the data of the test images
            tickets: Object that maps each filename of a ticket image with the
                corresponding EuromillionsTicket object
    """

    TEST_DATA_FOLDER = 'test/data'
    TEST_DATA_FILE = 'data.json'

    # Since Tesseract is not 100% reliable, this number establishes the
    # maximum differences that should exist between the actual numbers in
    # the ticket and the ones obtained with the scanner
    MAX_DIFFERENCE = 1

    def setUp(self):
        """
            Loads the JSON with the test data and scans each test image
        """

        # Load JSON
        data_file_path = os.path.join(self.TEST_DATA_FOLDER, self.TEST_DATA_FILE)
        data_file = open(data_file_path)
        self.data_json = json.load(data_file)

        # Scan each ticket
        self.tickets = {}
        for ticket in self.data_json['tickets']:
            filename = ticket['filename']
            file_path = os.path.join(self.TEST_DATA_FOLDER, filename)
            self.tickets[filename] = euro.Ticket(file_path)

    def test_number_sets(self):
        total_diffs = 0

        for ticket in self.data_json['tickets']:
            filename = ticket['filename']

            diffs = compare_sets(ticket['numbers'],
                                 self.tickets[filename].get_numbers())
            self.assertLessEqual(diffs, TicketParsingTest.MAX_DIFFERENCE)
            total_diffs += diffs

            diffs = compare_sets(ticket['stars'],
                                 self.tickets[filename].get_stars())
            self.assertLessEqual(diffs, TicketParsingTest.MAX_DIFFERENCE)
            total_diffs += diffs

        print 'Total differences:', total_diffs

    def test_text_parsing(self):
        text = textwrap.dedent('''\
            11N101418 30 44
            E04 05
            2.N 26 31 32 48 47
            E02 05
            ''')

        numbers = [[10, 14, 18, 30, 44],
                   [26, 31, 32, 48, 47]]

        stars = [[4, 5],
                 [2, 5]]

        parsed_numbers = euro.Ticket.parse_numbers(text, euro.Ticket.NUMBERS_REGEX)
        parsed_stars = euro.Ticket.parse_numbers(text, euro.Ticket.STARS_REGEX)

        self.assertEqual(numbers, parsed_numbers)
        self.assertEqual(stars, parsed_stars)


def compare_sets(arr1, arr2):
    """
        Method used to obtain the number of errors in the numbers returned
        by the scanner.
    """
    num_diffs = 0

    for i in range(len(arr1)):
        num_diffs += len(set(arr1[i]) - set(arr2[i]))

    return num_diffs


if __name__ == '__main__':
    nose.main()
