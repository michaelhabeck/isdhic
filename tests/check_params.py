from isdhic.params import Parameter
import gc

print id(posterior.params)
for f in posterior:
    print id(posterior.params)

print
print id(posterior.params['coordinates'])
for f in posterior.likelihoods:
    if hasattr(f.mock, '_coords'):
        print id(f.mock._coords)

params = []

for obj in gc.get_objects():
    if isinstance(obj, Parameter):
        params.append(obj)

for param in params:
    print id(param), param
