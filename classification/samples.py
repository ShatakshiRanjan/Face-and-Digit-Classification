import util
import zipfile
import os

## Constants
DATUM_WIDTH = 0  # in pixels
DATUM_HEIGHT = 0  # in pixels

## Module Classes

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.
    """

    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        global DATUM_WIDTH, DATUM_HEIGHT
        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data is None:
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        self.pixels = util.arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def getAsciiString(self):
        """
        Renders the data item as an ascii image.
        """
        rows = []
        data = util.arrayInvert(self.pixels)
        for row in data:
            ascii = list(map(asciiGrayscaleConversionFunction, row))
            rows.append("".join(ascii))
        return "\n".join(rows)

    def __str__(self):
        return self.getAsciiString()


# Data processing, cleanup, and display functions

def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less than n items if the end of the file is encountered).
    """
    global DATUM_WIDTH, DATUM_HEIGHT
    DATUM_WIDTH = width
    DATUM_HEIGHT = height
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < DATUM_WIDTH - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, DATUM_WIDTH, DATUM_HEIGHT))
    return items


def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).decode('utf-8').split('\n')


def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def asciiGrayscaleConversionFunction(value):
    """
    Helper function for display purposes.
    """
    if value == 0:
        return ' '
    elif value == 1:
        return '+'
    elif value == 2:
        return '#'


def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if character == ' ':
        return 0
    elif character == '+':
        return 1
    elif character == '#':
        return 2


def convertToInteger(data):
    """
    Helper function for file reading.
    """
    if isinstance(data, list):
        return list(map(convertToInteger, data))
    else:
        return IntegerConversionFunction(data)

# Testing

def _test():
    import doctest
    doctest.testmod()  # Test the interactive sessions in function comments
    n = 1
    #  items = loadDataFile("facedata/facedatatrain", n,60,70)
    #  labels = loadLabelsFile("facedata/facedatatrainlabels", n)
    items = loadDataFile("digitdata/trainingimages", n, 28, 28)
    labels = loadLabelsFile("digitdata/traininglabels", n)
    for i in range(1):
        print(items[i])
        print(items[i])
        print(items[i].height)
        print(items[i].width)
        print(dir(items[i]))
        print(items[i].getPixels())


if __name__ == "__main__":
    _test()
