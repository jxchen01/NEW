require 'nn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--modelPath','/home/jchen16/NEW/code/checkpoint/net_100.000000.bin','path to the trained model')
cmd:option('--imageDir', '/home/jchen16/NEW/data/temporal/raw', 'the directory to load')
cmd:option('--targetDir','/home/jchen16/NEW/data/temporal/mask','directory to the mask')
cmd:option('--outputDir', '/home/jchen16/NEW/data/temporal/encoder', 'the directory to save the input for RNN')
cmd:option('--segDir','/home/jchen16/NEW/data/temporal/prob','the directoty to save unet seg results')
cmd:option('--ext','.png','only load a specific type of images')
cmd:option('--training',false,'training mode or not')
cmd:option('--RAM',false,'load to RAM or not')
cmd:option('--XX',10,'the key parameter to determine the size of image')
cmd:option('--rho',3,'the length of sequence in RNN')
cmd:option('--gpu',1,'the gpu to use')
cmd:text()
opt = cmd:parse(arg or {})

XX=opt.XX
cutorch.setDevice(opt.gpu)

-- 1. Get the list of files in the given directory
if not opt.training then

    files = {}
    for file in paths.files(opt.imageDir) do
        if file:find(opt.ext .. '$') then
            table.insert(files, paths.concat(opt.imageDir,file))
        end
    end

    if #files == 0 then
	   error('given directory does not contain any files of type: ' .. opt.ext)
    end

    table.sort(files, function (a,b) return a < b end)
    numFrame = #files
else
    files={}
    files_target = {}
    for file in paths.files(opt.targetDir) do
        if file:find(opt.ext .. '$') then
            table.insert(files_target, paths.concat(opt.targetDir,file))
            table.insert(files, paths.concat(opt.imageDir,file))
        end
    end
    
    if #files == 0 then
        error('given directory does not contain any files of type: ' .. opt.ext)
    end

    table.sort(files_target, function (a,b) return a < b end)
    table.sort(files, function (a,b) return a < b end)
    numFrame = #files_target
end


-- 2. Load all the files into RAM
if opt.RAM then
    -- "images" is a table of tensors of size 1 x L x L 
    images = {}
    for i,file in ipairs(files) do
        table.insert(images, image.load(file))
    end
    targets={}
    for i,file in ipairs(files_target) do
        table.insert(targets,image.load(file,1,'byte'))
    end
end

-- 3. Load Model
unet=torch.load(opt.modelPath)
unet:evaluate()

-- 4. Define post-processing layers
softmax = nn.SoftMax()
reshape_back = nn.Reshape((16*XX-92),(16*XX-92),2)

-- 5. Process images one by one 
pad = nn.SpatialZeroPadding(92,92,92,92) --the padding filter (92 is fixed, no need to change)
local fm_table={}

for i=1, #files do
	-- padding 
    local image_whole, raw
    if opt.RAM then 
        raw = images[i]
    else
        raw = image.load(files[i])
    end
    image_whole = pad:forward(raw)

	-- build tiles
	local tiles = {} 
    local dd=16*XX-92
    local windowSize = 16*XX+92
    local numX=math.ceil(raw:size(2)/dd);
    local numY=math.ceil(raw:size(3)/dd);

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
    local tile_output=torch.Tensor(#tiles,dd,dd)
    local tile_fm=torch.Tensor(#tiles,64,dd,dd)
    for ti=1,#tiles do
        local b=unet:forward(tiles[ti]:cuda()):double()
        local c=softmax:forward(b)
        local d=reshape_back:forward(c)
        local ff=d:select(3,2)
        tile_output[ti]:copy(ff)  -- cell has label 2   

        local fm=unet.modules[66].output
        tile_fm[ti]:copy(fm)
    end

    
    -- assemble back to the whole image
    local output_image = torch.Tensor(raw:size(2),raw:size(3))
    local output_fm=torch.Tensor(64,raw:size(2),raw:size(3))
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
    		output_image:sub(x1, x2, y1, y2):copy(tile_output:select(1,tile_idx):sub(dd-(x2-x1),dd,dd-(y2-y1),dd))
            output_fm:sub(1,64,x1,x2,y1,y2):copy(tile_fm:select(1,tile_idx):sub(1,64,dd-(x2-x1),dd,dd-(y2-y1),dd))
    	end
    end
    
    -- write the segmentation result to file
    local filename=string.format('%s/prob_%d.png',opt.segDir,i);
    image.save(filename,output_image)
     
    -- prepare for the data for temporal RNN
    table.insert(fm_table, output_fm) -- feature map
end

if opt.training then
    if not opt.RAM then
        targets={}
        for i,file in ipairs(files_target) do
            table.insert(targets,image.load(file,1,'byte'))
        end
    end
    -- compute the tiles
    local dd=16*XX-92
    local xdim = images[1]:size(2)
    local ydim = images[1]:size(3)
    local numX=math.ceil(xdim/dd);
    local numY=math.ceil(ydim/dd);
    print(numX)
    print(numY)

    local data_idx=0

    for xi=1,numX do
        local x1,x2,y1,y2
        
        if xi==numX then
            x2=xdim
            x1=xdim-dd+1
        else
            x1=1+(xi-1)*dd
            x2=xi*dd
        end

        for yi=1,numY do
            
            if yi==numY then
                y2=ydim
                y1=ydim-dd+1
            else      
                y1=1+(yi-1)*dd      
                y2=yi*dd
            end          

            print('processing '..xi..', '..yi)
            
            local input_table = {} 
            local target_table ={}
            local init_fm = {}
            for ti=1, #files-opt.rho do
                table.insert(init_fm, targets[ti]:sub(1,1,x1,x2,y1,y2))
            end

            for ti=2, #files do
                table.insert(input_table, fm_table[ti]:sub(1,64,x1,x2,y1,y2))
                table.insert(target_table, torch.reshape(targets[ti]:sub(1,1,x1,x2,y1,y2),dd*dd,1))
            end

            data_idx = data_idx + 1
            obj={input=input_table, target=target_table, init=init_fm}
            local str= string.format('%s/train_%d.t7',opt.outputDir,data_idx);
            torch.save(str,obj)
        end

        collectgarbage()
    end

end

