require 'nn'
require 'nngraph'

XX=30

input = nn.Identity()()

L1=nn.Sequential()
L1:add(nn.SpatialConvolution(1, 64, 3, 3, 1, 1, 0, 0))
L1:add(nn.ReLU())
L1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0))
L1:add(nn.ReLU())

L2=nn.Sequential()
L2:add(nn.SpatialMaxPooling(2, 2, 2, 2)(L1))
L2:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0))
L2:add(nn.ReLU())
L2:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0))
L2:add(nn.ReLU())

L3=nn.Sequential()
L3:add(nn.SpatialMaxPooling(2, 2, 2, 2)(L2))
L3:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0))
L3:add(nn.ReLU())
L3:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0))
L3:add(nn.ReLU())

L4=nn.Sequential()
L4:add(nn.SpatialMaxPooling(2, 2, 2, 2)(L3))
L4:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0))
L4:add(nn.ReLU())
L4:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0))
L4:add(nn.ReLU())

L5=nn.Sequential()
L5:add(nn.SpatialMaxPooling(2, 2, 2, 2)(L4))
L5:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0))
L5:add(nn.ReLU())
L5:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0))
L5:add(nn.ReLU())

local offset = 4
local length = 2*XX-4
L4C=nn.Narrow(3,offset,length)(nn.Narrow(2,offset,length)(L4))
L5up=nn.SpatialFullConvolution(1024, 512, 2, 2, 2, 2)(L5)


L6=nn.Sequential()
L6:add(nn.JoinTable(1)({L5up,L4C}))
L6:add(nn.SpatialConvolution(1024,512, 3, 3, 1, 1, 0, 0))
L6:add(nn.ReLU())
L6:add(nn.SpatialConvolution(512,512, 3, 3, 1, 1, 0, 0))
L6:add(nn.ReLU())

local offset = 16
local length = 4*XX-16
L3C=nn.Narrow(3,offset,length)(nn.Narrow(2,offset,length)(L3))
L6up=nn.SpatialFullConvolution(512, 256, 2, 2, 2, 2)(L6)

L7=nn.Sequential()
L7:add(nn.JoinTable(1)({L6up,L3C}))
L7:add(nn.SpatialConvolution(512,256, 3, 3, 1, 1, 0, 0))
L7:add(nn.ReLU())
L7:add(nn.SpatialConvolution(256,256, 3, 3, 1, 1, 0, 0))
L7:add(nn.ReLU())


local offset = 40
local length = 8*XX-40
L2C=nn.Narrow(3,offset,length)(nn.Narrow(2,offset,length)(L2))
L7up=nn.SpatialFullConvolution(256, 128, 2, 2, 2, 2)(L7)

L8=nn.Sequential()
L8:add(nn.JoinTable(1)({L7up,L2C}))
L8:add(nn.SpatialConvolution(256,128, 3, 3, 1, 1, 0, 0))
L8:add(nn.ReLU())
L8:add(nn.SpatialConvolution(128,128, 3, 3, 1, 1, 0, 0))
L8:add(nn.ReLU())

local offset = 88
local length = 16*XX-88
L1C=nn.Narrow(3,offset,length)(nn.Narrow(2,offset,length)(L1))
L8up=nn.SpatialFullConvolution(128, 64, 2, 2, 2, 2)(L8)

L9=nn.Sequential()
L9:add(nn.JoinTable(1)({L8up,L1C}))
L9:add(nn.SpatialConvolution(128,64, 3, 3, 1, 1, 0, 0))
L9:add(nn.ReLU())
L9:add(nn.SpatialConvolution(64,64, 3, 3, 1, 1, 0, 0))
L9:add(nn.ReLU())

L10=nn.Sequential()
L10:add(nn.SpatialConvolution(64, 1, 1, 1, 1, 1, 1, 0, 0)(L9))

unet = nn.gModule({input},{L10})

input_image = torch.rand(1,572,572)
label_image = torch.rand(388,388)

unet:forward(input_image)
unet:backward(input_image, label_image)

graph.dot(unet.fg, 'Forward_Graph')
graph.dot(unet.bg, 'Backward_Graph')
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
