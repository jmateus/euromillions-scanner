import unittest
import nose
import os
import json
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
        for ticket in self.data_json['tickets']:
            filename = ticket['filename']

            self.assertEqual(ticket['numbers'],
                             self.tickets[filename].get_numbers(),
                             '%s has the correct numbers' % filename)

            self.assertEqual(ticket['stars'],
                             self.tickets[filename].get_stars(),
                             '%s has the correct stars' % filename)


if __name__ == '__main__':
    nose.main()
