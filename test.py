def add_dict(d, k, v):
	d[k] = v

class A:
	def __init__(self):
		self.a = 1

	def new(self):
		self.b = 2


if  __name__ == "__main__":
	d = dict({2:2})
	add_dict(d, 1, 1)
	print(d)

	a = A()
	a.new()
	print(a.b)

