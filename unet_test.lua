require 'nn'
require 'nngraph'

require 'cutorch'
require 'cunn'
cutorch.setDevice(2)


local freeMemory, totalMemory

XX=30

input_image = torch.rand(1,572,572)
label_image = torch.Tensor(388*388,1):random(1,2)

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)



input = nn.Identity()()

L1a=nn.SpatialConvolution(1, 64, 3, 3, 1, 1, 0, 0)(input)
L1b=nn.ReLU(true)(L1a)
L1c=nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0)(L1b)
L1=nn.ReLU(true)(L1c)

L2a=nn.SpatialMaxPooling(2, 2, 2, 2)(L1)
L2b=nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0)(L2a)
L2c=nn.ReLU(true)(L2b)
L2d=nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0)(L2c)
L2=nn.ReLU(true)(L2d)

L3a=nn.SpatialMaxPooling(2, 2, 2, 2)(L2)
L3b=nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0)(L3a)
L3c=nn.ReLU(true)(L3b)
L3d=nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0)(L3c)
L3=nn.ReLU(true)(L3d)

L4a=nn.SpatialMaxPooling(2, 2, 2, 2)(L3)
L4b=nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0)(L4a)
L4c=nn.ReLU(true)(L4b)
L4d=nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0)(L4c)
L4=nn.ReLU(true)(L4d)

L5a=nn.SpatialMaxPooling(2, 2, 2, 2)(L4)
L5b=nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0)(L5a)
L5c=nn.ReLU(true)(L5b)
L5d=nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0)(L5c)
L5=nn.ReLU(true)(L5d)

Crop4=nn.Narrow(2,4,2*XX-4)(L4)
L4cp=nn.Narrow(3,4,2*XX-4)(Crop4)
L5up=nn.SpatialFullConvolution(1024, 512, 2, 2, 2, 2)(L5)

L6a=nn.JoinTable(1,3)({L5up,L4cp})
L6b=nn.SpatialConvolution(1024,512, 3, 3, 1, 1, 0, 0)(L6a)
L6c=nn.ReLU(true)(L6b)
L6d=nn.SpatialConvolution(512,512, 3, 3, 1, 1, 0, 0)(L6c)
L6=nn.ReLU(true)(L6d)

Crop3=nn.Narrow(2,16,104)(L3)
L3cp=nn.Narrow(3,16,104)(Crop3)
L6up=nn.SpatialFullConvolution(512, 256, 2, 2, 2, 2)(L6)

L7a=nn.JoinTable(1,3)({L6up,L3cp})
L7b=nn.SpatialConvolution(512,256, 3, 3, 1, 1, 0, 0)(L7a)
L7c=nn.ReLU(true)(L7b)
L7d=nn.SpatialConvolution(256,256, 3, 3, 1, 1, 0, 0)(L7c)
L7=nn.ReLU(true)(L7d)

Crop2=nn.Narrow(2,40,200)(L2)
L2cp=nn.Narrow(3,40,200)(Crop2)
L7up=nn.SpatialFullConvolution(256, 128, 2, 2, 2, 2)(L7)

L8a=nn.JoinTable(1,3)({L7up,L2cp})
L8b=nn.SpatialConvolution(256,128, 3, 3, 1, 1, 0, 0)(L8a)
L8c=nn.ReLU(true)(L8b)
L8d=nn.SpatialConvolution(128,128, 3, 3, 1, 1, 0, 0)(L8c)
L8=nn.ReLU(true)(L8d)

Crop1=nn.Narrow(2,88,392)(L1)
L1cp=nn.Narrow(3,88,392)(Crop1)
L8up=nn.SpatialFullConvolution(128, 64, 2, 2, 2, 2)(L8)

L9a=nn.JoinTable(1,3)({L8up,L1cp})
L9b=nn.SpatialConvolution(128,64, 3, 3, 1, 1, 0, 0)(L9a)
L9c=nn.ReLU(true)(L9b)
L9d=nn.SpatialConvolution(64,64, 3, 3, 1, 1, 0, 0)(L9c)
L9=nn.ReLU(true)(L9d)

L10a=nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(L9)
L10b=nn.Transpose({1,2},{2,3})(L10a)
L10=nn.Reshape(388*388,2)(L10b)

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)

unet = nn.gModule({input},{L10})
criterion = nn.CrossEntropyCriterion()

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)

--[[
local finput, fgradInput
unet:apply(function(m) if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'nn.SpatialFullConvolution' then 
                           finput = finput or m.finput
                           fgradInput = fgradInput or m.fgradInput
                           m.finput = finput
                           m.fgradInput = fgradInput
                        end
            end)
--]]

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)


collectgarbage()

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)



collectgarbage()

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)

output_image = unet:forward(input_image)

freeMemory, totalMemory = cutorch.getMemoryUsage(2)
print(freeMemory)
print(totalMemory)


local err = criterion:forward(output_image, label_image)
local gradCriterion = criterion:backward(output_image, label_image)
unet:zeroGradParameters()
unet:backward(input_image,gradCriterion)
unet:updateParameters(0.05)

print(err)

