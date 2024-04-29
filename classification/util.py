# util.py
# -------
# Licensing Information: You are free to use and extend these projects for educational purposes. 
# The Pacman AI projects were developed at UC Berkeley, primarily by John DeNero (denero@cs.berkeley.edu) 
# and Dan Klein (klein@cs.berkeley.edu). For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import sys
import inspect
import heapq
import random
import signal

class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."
    def __init__(self):
        self.list = []
    
    def push(self, item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []
  
    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.append(item)

    def pop(self):
        "Dequeue the earliest enqueued item still in the queue."
        return self.list.pop(0)

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    def __init__(self):  
        self.heap = []
    
    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item
  
    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    def __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction
        super().__init__()
    
    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        super().push(item, self.priorityFunction(item))

def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class Counter(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self.keys()) == 0: return None
        all_items = list(self.items())
        values = [x[1] for x in all_items]
        maxIndex = values.index(max(values))
        return all_items[maxIndex][0]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def __mul__(self, y):
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]      
        return sum

    def __radd__(self, y):
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

def raiseNotDefined():
    print("Method not implemented: %s" % inspect.stack()[1][3])
    sys.exit(1)

def normalize(vectorOrCounter):
    normalizedCounter = Counter()
    if isinstance(vectorOrCounter, Counter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0: return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]

def flipCoin(p):
    return random.random() < p

def nearestPoint(pos):
    (current_row, current_col) = pos
    grid_row = int(current_row + 0.5)
    grid_col = int(current_col + 0.5)
    return (grid_row, grid_col)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

## code to handle timeouts
class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass

class TimeoutFunction:
    def __init__(self, function, timeout):
        "timeout must be at least 1 second. WHY??"
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if not 'SIGALRM' in dir(signal):
            return self.function(*args)
        old = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        try:
            result = self.function(*args)
        finally:
            signal.signal(signal.SIGALRM, old)
        signal.alarm(0)
        return result
