# Problem Solving with Algorithms and Data Structures
# Brad Miller, David Ranum


class Queue:
    def __init__(self):
        self.contents = []

    def enqueue(self, item):
        self.contents.insert(0, item)

    def dequeue(self):
        return self.contents.pop()

    def peek(self):
        return self.contents[-1]

    def is_empty(self):
        return len(self.contents) == 0

    def size(self):
        return len(self.contents)

    def __str__(self):
        return ', '.join(self.contents)


def josephus_problem(person_list, num):
    '''
    Who will survive if we play the Josephus game on
    all the people in person_list?
    '''

    q = Queue()
    for p in person_list:
        q.enqueue(p)

    # continue until only one person is left alive
    while q.size() > 1:
        # go around the circle
        for i in range(num):
            q.enqueue(q.dequeue())

        # permanently remove one person
        q.dequeue()

    # who is left alive?
    return q.dequeue()


print(josephus_problem(["Bill", "David", "Susan", "Jane", "Kent",
    "Brad"], 7))
