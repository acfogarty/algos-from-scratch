# Problem Solving with Algorithms and Data Structures
# Brad Miller, David Ranum


class Stack:
    def __init__(self):
        self.contents = []

    def push(self, item):
        self.contents.append(item)

    def pop(self):
        return self.contents.pop()

    def peek(self):
        return self.contents[-1]

    def is_empty(self):
        return len(self.contents) == 0

    def __str__(self):
        return ', '.join(self.contents)


def detect_balanced_parentheses(string):
    '''
    Returns:
        boolean (True if parentheses are balanced in input string)
    '''
    s = Stack()
    balanced = True
    for c in string:
        if c == '(':
            s.push(c)
        elif c == ')':
            if s.is_empty():
                balanced = False
            else:
                s.pop()

    balanced = balanced and s.is_empty()

    print(string)
    print(balanced)

    return balanced


def decimal_to_binary(integer):
    s = Stack()
    while integer > 0:
        remainder = integer % 2
        s.push(remainder)
        integer = integer // 2

    binary = []
    while not s.is_empty():
        binary.append(str(s.pop()))

    return int(''.join(binary))


string = '((()()(())))'
detect_balanced_parentheses(string)


integer = 233
print(decimal_to_binary(integer))
