require 'nn'

local channel_down={64,128,256,512}
local channel_up={512,256,128,64}
local input_channel=1
local conv_kernel = 3


unet = nn.Sequential()

-- contracting path 
for i, output_channel in ipairs(channel_down) do

	unet:add(nn.SpatialConvolution(input_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())
	unet:add(nn.SpatialConvolution(output_channel,output_channel, conv_kernel, conv_kernel, 1, 1, 0, 0))
	unet:add(nn.ReLU())
	unet:add(nn.SpatialMaxPooling(2,2,2,2))

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




