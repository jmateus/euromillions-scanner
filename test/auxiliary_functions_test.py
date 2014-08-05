import unittest
import nose
from euromillions_scanner import scanner


class EnclosedRectangleTest(unittest.TestCase):
    def test_different_widths(self):
        rect1 = (1, 6, 11, 5)
        rect2 = (3, 15, 15, 15)
        enclosed_rect = (1, 11, 17, 4)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)

    def test_overlapping_width(self):
        rect1 = (6, 6, 4, 5)
        rect2 = (3, 15, 15, 15)
        enclosed_rect = (3, 11, 15, 4)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)

        self.assertEqual(scanner.get_enclosed_rectangle(rect2, rect1),
                         enclosed_rect)


class ResizeFunctionTest(unittest.TestCase):
    def _ratio(self, (width, height)):
        return float(width) / height

    def test_inversed_max_size(self):
        size = (200, 250)
        max_size = (250, 200)
        new_size = scanner.resize(size, max_size)

        self.assertEqual(self._ratio(size), self._ratio(new_size))

    def test_same_size(self):
        size = (200, 250)
        max_size = (200, 250)
        new_size = scanner.resize(size, max_size)

        self.assertEqual(size, new_size)

    def test_resize_to_smaller_dimensions(self):
        size = (200, 250)
        max_size = (100, 100)
        new_size = scanner.resize(size, max_size)

        self.assertEqual(self._ratio(size), self._ratio(new_size))
        self.assertEqual(max(new_size), max(max_size))


if __name__ == '__main__':
    nose.main()
