# EuroMillions Scanner

A Python script to scan portuguese EuroMillions tickets and extract the numbers. This is merely a POC and (very) far from perfect.


## Usage

```python
import euromillions_scanner as euro

t = euro.Ticket('path_to_ticket_image.jpg')

# Get the main numbers of the ticket
t.get_main_numbers()

# Get the lucky stars
t.get_stars()

# Get the first set of the ticket (pair of main numbers and stars)
t[0]
```


## Requirements

* Python 2.7+
* Numpy 1.8+
* OpenCV 2.4+
* Python-Tesseract 0.9+


## Installation

* Install all the dependencies
* Clone the repository
* Download the Python-Tesseract sample code (http://python-tesseract.googlecode.com/files/test-slim.7z) and copy the `tessdata` folder to the `euromillions_scanner` folder (the one with the `scanner.py`)
* Run `python setup.py install`


## License
MIT


## TODO
* Check if the ticket has any prizes
* Improve the installation process
* Add the ability to crop the ticket and remove the background
* Add more tests
