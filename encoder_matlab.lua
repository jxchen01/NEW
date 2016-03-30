require 'nn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
require 'cudnn'
matio=require 'matio'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--modelPath','/home/jchen16/NEW/code/checkpoint/fungus_XX_20_seq_whole/unet_420000.000000.bin','path to the trained model')
cmd:option('--imageDir', '/home/jchen16/NEW/data/fungus/training_data', 'the directory to load')
cmd:option('--outputDir', '/home/jchen16/NEW/data/fungus/encoder', 'the directory to save the input for RNN')
cmd:option('--ext','.mat','only load a specific type of images')
cmd:option('--training',false,'training mode or not')
cmd:option('--RAM',false,'load to RAM or not')
cmd:option('--XX',20,'the key parameter to determine the size of image')
cmd:option('--rho',3,'the length of sequence in RNN')
cmd:option('--gpu',1,'the gpu to use')
cmd:option('--outputLayer',2,'1 means the last one, 2 means the second last one')
cmd:option('--imageType',1,'1: grayscale, 3: RGB')
cmd:text()
opt = cmd:parse(arg or {})

XX=opt.XX
cutorch.setDevice(opt.gpu)

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

if opt.RAM then
-- 2. Load all the files into RAM
   data = {}
   for i,file in ipairs(files) do
      -- load each image
      table.insert(data, matio.load(file))
   end
end

-- 3. Load Model
unet=torch.load(opt.modelPath)
unet:evaluate()

-- 4. Process images one by one 
for i=1, #files do
	
    local image_seq, target_seq
    if opt.RAM then
        image_seq = data[i].image
        if opt.training then
            target_seq = data[i].target
        end
    else
        local data = matio.load(files[i])
        image_seq = data.image
        if opt.training then
            target_seq = data.target
        end
    end

    local fm_seq={}

	for j=1,#image_seq do
        print('i='..i..', j='..j)

        local input_image = torch.FloatTensor(1,opt.imageType,16*XX+92,16*XX+92)
        input_image[1][1]=image_seq[j]
	
        local b=unet:forward(input_image:cuda()):double()
        local fm
        if opt.outputLayer==1 then
            fm = b[1]
        elseif opt.outputLayer==2 then
            fm=unet.modules[66].output[1]
        end
        table.insert(fm_seq, fm:float())
        print(fm:size())
        
        if i==1 and j<10 then
            local softmax = nn.SoftMax()
            out = softmax:forward(b[1])
            local str0= string.format('test%d.png',j);
            image.save(str0,out:select(1,4))
            --[[
            for kk=1,fm:size(1) do
                local a=fm:select(1,kk)
                print(a[1][100])
                print(type(a))
                local str1= string.format('test_fm_%d.png',kk);
                image.save(str1,a:float())
            end
            --]]
        end
        
    end

    if opt.training then
        obj={input=fm_seq, target=target_seq}
    else
        obj={input=fm_seq}
    end
    local str= string.format('%s/train_%d.t7',opt.outputDir,i);
    torch.save(str,obj)

    collectgarbage()
    -- write the segmentation result to file
    --local filename=string.format('%s/prob_%d.png',opt.segDir,i);
    --image.save(filename,output_image)
end

