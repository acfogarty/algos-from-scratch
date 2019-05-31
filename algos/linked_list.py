# Problem Solving with Algorithms and Data Structures
# Brad Miller, David Ranum


class Node:

    def __init__(self, data):
        self.data = data
        self.next_node = None

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_next(self):
        return self.next_node

    def set_next(self, next_node):
        self.next_node = next_node

    def __str__(self):
        if self.next_node != None:
            next_data = self.next_node.get_data()
        else:
            next_data = None
        return 'data = {}, next = {}'.format(self.data,
                                             next_data)


class UnorderedList:

    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head == None

    def add(self, item):
        n = Node(item)
        n.set_next(self.head)
        self.head = n

    def size(self):
        count = 0
        current = self.head
        while current != None:
            count += 1
            current = current.get_next()

        return count

    def search(self, item):
        current = self.head
        found = False
        while (not found) and (current != None):
            if current.get_data() != item:
                found = True
            else:
                current = current.get_next()

        return found

    def __str__(self):
        string = []
        current = self.head
        while current != None:
            string.append(str(current))
            current = current.get_next()
        return '\n'.join(string)


ul = UnorderedList()
ul.add(5)
ul.add(7)
ul.add(9)
print(ul)
