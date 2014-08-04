import unittest
import nose
from euromillions_scanner import scanner


class AuxiliaryFunctionsTest(unittest.TestCase):
    def test_enclosed_rectangle(self):
        rect1 = (1, 6, 11, 5)
        rect2 = (3, 15, 15, 15)
        enclosed_rect = (1, 11, 17, 4)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)

    def test_enclosed_rectangle_overlapping_width(self):
        rect1 = (6, 6, 4, 5)
        rect2 = (3, 15, 15, 15)
        enclosed_rect = (3, 11, 15, 4)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)


if __name__ == '__main__':
    nose.main()
