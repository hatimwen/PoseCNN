class A(object):
    def __init__(self):
        print("Parent")
    def wow(self):
        print("wow")

class B(A):
    def __init__(self):
        # A.__init__(self)
        print("Child")
        super(B, self).__init__()

c = B()
c.wow()