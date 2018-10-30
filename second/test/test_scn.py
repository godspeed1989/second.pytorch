import torch
import sparseconvnet as scn

# Use the GPU if there is one, otherwise CPU
use_gpu = torch.cuda.is_available()

model = scn.Sequential().add(
    scn.SparseVggNet(dimension=2, nInputPlanes=1,
		     layers=[['C',  8], ['C',  8], ['MP', 3, 2],
		             ['C', 16], ['C', 16], ['MP', 3, 2],
		             ['C', 24], ['C', 24], ['MP', 3, 2]])
).add(
    scn.SubmanifoldConvolution(dimension=2, nIn=24, nOut=32, filter_size=3, bias=False)
).add(
    scn.BatchNormReLU(nPlanes=32)
).add(
    scn.SparseToDense(dimension=2,nPlanes=32)
)
if use_gpu:
    model.cuda()

# output will be 10x10, calculate the input size
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
print(inputSpatialSize) # (87, 87)
#input = scn.InputBatch(dimension=2, spatial_size=inputSpatialSize)
input_scn = scn.InputLayer(dimension=2, spatial_size=inputSpatialSize)

msg = [
    " X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
    " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
    " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
    " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
    " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "]

#Add a sample using set_locations
#input.add_sample()
locations = []
features = []
for y, line in enumerate(msg):
    for x, c in enumerate(line):
        if c == 'X':
            locations.append([x,y,0])
            features.append([1])
locations = torch.LongTensor(locations)
print('locations.size', locations.size())   # ([101, 3])
features = torch.FloatTensor(features)
print('features.size', features.size())     # ([101, 1])

ret = input_scn.forward((locations, features, 1))

model.train()
if use_gpu:
    ret.cuda()
output = model.forward(ret)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print(output.size(), output.type())
