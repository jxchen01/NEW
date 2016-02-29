require 'nn'
require 'nngraph'

require 'cutorch'
require 'cunn'
cutorch.setDevice(2)

XX=30

input = nn.Identity()()

L1S=nn.Sequential()
L1S:add(nn.SpatialConvolution(1, 64, 3, 3, 1, 1, 0, 0))
L1S:add(nn.ReLU())
L1S:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0))
L1S:add(nn.ReLU())
L1=L1S(input)

L2S=nn.Sequential()
L2S:add(nn.SpatialMaxPooling(2, 2, 2, 2))
L2S:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0))
L2S:add(nn.ReLU())
L2S:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0))
L2S:add(nn.ReLU())
L2=L2S(L1)

L3S=nn.Sequential()
L3S:add(nn.SpatialMaxPooling(2, 2, 2, 2))
L3S:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0))
L3S:add(nn.ReLU())
L3S:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0))
L3S:add(nn.ReLU())
L3=L3S(L2)

L4S=nn.Sequential()
L4S:add(nn.SpatialMaxPooling(2, 2, 2, 2))
L4S:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0))
L4S:add(nn.ReLU())
L4S:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0))
L4S:add(nn.ReLU())
L4=L4S(L3)

L5S=nn.Sequential()
L5S:add(nn.SpatialMaxPooling(2, 2, 2, 2))
L5S:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0))
L5S:add(nn.ReLU())
L5S:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0))
L5S:add(nn.ReLU())
L5=L5S(L4)

Crop4=nn.Sequential()
Crop4:add(nn.Narrow(2,4,56)) 
Crop4:add(nn.Narrow(3,4,56)) 
L4C=Crop4(L4)
L5up=nn.SpatialFullConvolution(1024, 512, 2, 2, 2, 2)(L5)


L6S=nn.Sequential()
L6S:add(nn.JoinTable(1,3)) 
L6S:add(nn.SpatialConvolution(1024,512, 3, 3, 1, 1, 0, 0))
L6S:add(nn.ReLU())
L6S:add(nn.SpatialConvolution(512,512, 3, 3, 1, 1, 0, 0))
L6S:add(nn.ReLU())
L6=L6S({L5up,L4C})

Crop3=nn.Sequential()
Crop3:add(nn.Narrow(2,16,104))
Crop3:add(nn.Narrow(3,16,104))
L3C=Crop3(L3)
L6up=nn.SpatialFullConvolution(512, 256, 2, 2, 2, 2)(L6)

L7S=nn.Sequential()
L7S:add(nn.JoinTable(1,3)) 
L7S:add(nn.SpatialConvolution(512,256, 3, 3, 1, 1, 0, 0))
L7S:add(nn.ReLU())
L7S:add(nn.SpatialConvolution(256,256, 3, 3, 1, 1, 0, 0))
L7S:add(nn.ReLU())
L7=L7S({L6up,L3C})

Crop2=nn.Sequential()
Crop2:add(nn.Narrow(2,40,200))
Crop2:add(nn.Narrow(3,40,200))
L2C=Crop2(L2)
L7up=nn.SpatialFullConvolution(256, 128, 2, 2, 2, 2)(L7)

L8S=nn.Sequential()
L8S:add(nn.JoinTable(1,3)) 
L8S:add(nn.SpatialConvolution(256,128, 3, 3, 1, 1, 0, 0))
L8S:add(nn.ReLU())
L8S:add(nn.SpatialConvolution(128,128, 3, 3, 1, 1, 0, 0))
L8S:add(nn.ReLU())
L8=L8S({L7up,L2C})

Crop1=nn.Sequential()
Crop1:add(nn.Narrow(2,88,392))
Crop1:add(nn.Narrow(3,88,392))
L1C=Crop1(L1)
L8up=nn.SpatialFullConvolution(128, 64, 2, 2, 2, 2)(L8)

L9S=nn.Sequential()
L9S:add(nn.JoinTable(1,3)) 
L9S:add(nn.SpatialConvolution(128,64, 3, 3, 1, 1, 0, 0))
L9S:add(nn.ReLU())
L9S:add(nn.SpatialConvolution(64,64, 3, 3, 1, 1, 0, 0))
L9S:add(nn.ReLU())
L9=L9S({L8up,L1C})


L10S=nn.Sequential()
L10S:add(nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0))
L10S:add(nn.Transpose({1,2},{2,3}))
L10S:add(nn.Reshape(388*388,2))
L10=L10S(L9)

unet = nn.gModule({input},{L10}):cuda()

local finput, fgradInput
unet:apply(function(m) if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'nn.SpatialConvolutionMM' then 
                           finput = finput or m.finput
                           fgradInput = fgradInput or m.fgradInput
                           m.finput = finput
                           m.fgradInput = fgradInput
                        end
            end)

input_image = torch.rand(1,572,572):cuda()
label_image = torch.Tensor(388*388,2):random(1,2):cuda()

collectgarbage()

output_image = unet:forward(input_image)
criterion = nn.CrossEntropyCriterion():cuda()

local err = criterion:forward(output_image, label_image)
local gradCriterion = criterion:backward(output_image, label_image)
unet:zeroGradParameters()
unet:backward(input_image,gradCriterion)
unet:updateParameters(0.05)

print(err)



--[[
-- contracting path 
for i, output_channel in ipairs(channel_down) do

	unet:add(nn.SpatialConvolution(input_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())
	unet:add(nn.SpatialConvolution(output_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())
	unet:

	input_channel = output_channel
end


-- vallay
unet:add(nn.SpatialConvolution(input_channel,1024, conv_kernel, conv_kernel, 1, 1, 0, 0))
unet:add(nn.ReLU())
unet:add(nn.SpatialConvolution(1024,1024, conv_kernel, conv_kernel, 1, 1, 0, 0))
unet:add(nn.ReLU())

input_channel=1024

-- expansion path 
for i, output_channel in ipairs(channel_up) do 
	unet:add(nn.SpatialFullConvolution(input_channel, output_channel, 2, 2, 2, 2))
	unet:add(nn.SpatialConvolution(output_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())
	unet:add(nn.SpatialConvolution(output_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())

	input_channel = output_channel
end



unet:add(nn.SpatialConvolution(output_channel,1, 1, 1, 1, 1, 0, 0))
--]]
