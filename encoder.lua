require 'nn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'

cutorch.setDevice(2)


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--modelPath','/home/jchen16/NEW/code/checkpoint/net_100.000000.bin','path to the trained model')
cmd:option('--imageDir', '/home/jchen16/NEW/data/sample/', 'the directory to load')
cmd:option('--outputDir', '/home/jchen16/NEW/data/prob/', 'the directory to load')
cmd:option('--ext','.png','only load a specific type of images')
cmd:option('--XX',10,'the key parameter to determine the size of image')
cmd:text()
opt = cmd:parse(arg or {})

XX=opt.XX

-- 1. Get the list of files in the given directory

files = {}

for file in paths.files(opt.imageDir) do
   if file:find(opt.ext .. '$') then
      table.insert(files, paths.concat(opt.imageDir,file))
   end
end

if #files == 0 then
	error('given directory doesnt contain any files of type: ' .. opt.ext)
end

table.sort(files, function (a,b) return a < b end)

-- 2. Load all the files into RAM
-- "images" is a table of tensors of size 1 x L x L 
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

-- 3. Load Model
unet=torch.load(opt.modelPath)
unet:evaluate()

-- 4. Define post-processing layers
softmax = nn.SoftMax()
reshape_back = nn.Reshape((16*XX-92),(16*XX-92),2)

-- 5. Process images one by one 

pad = nn.SpatialZeroPadding(92,92,92,92) --the padding filter

for i=1, #images do
	-- padding 
	local image_whole = pad:forward(images[i])

	-- build tiles
	local tiles = {} 
    local dd=16*XX-92
    local windowSize = 16*XX+92
    local numX=math.ceil(images[i]:size(2)/dd);
    local numY=math.ceil(images[i]:size(3)/dd);

    for xi=1,numX do
    	local x0, y0
    	if xi==numX then
    		x0=image_whole:size(2) - windowSize + 1
    	else
    		x0=1+dd*(xi-1)
    	end
    	for yi=1,numY do
    		if yi==numY then
    			y0=image_whole:size(3) - windowSize + 1
    		else
    			y0=1+dd*(yi-1)
    		end
    		table.insert(tiles, image_whole:sub(1,3,x0,x0+windowSize-1,y0,y0+windowSize-1))
    	end
    end

    -- process each tile
    local tile_output={}
    for ti=1,#tiles do
    	local b=unet:forward(tiles[ti]:cuda()):double()
    	local c=softmax:forward(b)
    	local d=reshape_back:forward(c)
    	table.insert(tile_output, d:select(3,2))  -- cell has label 2
    end

    -- assemble back to the whole image
    output_image = torch.Tensor(images[i]:size(2),images[i]:size(3))
    local tile_idx=0
    for xi=1,numX do
    	local x1,x2,y1,y2
    	x1=1+(xi-1)*dd
    	if xi==numX then
    		x2=images[i]:size(2)
    	else
    		x2=xi*dd
    	end
    	for yi=1,numY do
    		y1=1+(yi-1)*dd
    		if yi==numY then
    			y2=images[i]:size(3)
    		else			
    			y2=yi*dd
    		end
    		tile_idx=tile_idx+1
    		output_image:sub(x1, x2, y1, y2):copy(tiles[tile_idx]:sub(dd-(x2-x1),dd,dd-(y2-y1),dd))
    	end
    end
    
    -- write the result to file
    local filename=string.format('%s/prob_%f.png',opt.outputDir,i);
    image.save(filename,output_image)
end

--[[
a=image.load('/home/jchen16/NEW/data/X10/train/1.png')
a=a:cuda()
b=unet:forward(a)
b=b:double()

softmax = nn.SoftMax()
c=softmax(b)

reshape_back = nn.Reshape((16*XX-92),(16*XX-92),2)
d=reshape_back:forward(c)

output = d:select(3,2)
--]]