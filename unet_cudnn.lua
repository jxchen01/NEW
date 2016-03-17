require 'nn'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--imageDir', '/home/jchen16/NEW/data/X10/train/', 'the directory to load')
cmd:option('--labelDir', '/home/jchen16/NEW/data/X10/label/', 'the directory to load')
cmd:option('--ext','.png','only load a specific type of images')
cmd:option('--epoch',100,'the number of iterations trained on the whole dataset')
cmd:option('--learningRate',0.005,'initial learning rate')
cmd:option('--minLR',0.0005,'minimal learning rate')
cmd:option('--dropoutProb', 0.25, 'probability of zeroing a neuron (dropout probability)')
cmd:option('--uniform', 0.05, 'initialize parameters using uniform distribution between -uniform and uniform.')
cmd:option('--CheckPointDir', '/home/jchen16/NEW/code/checkpoint','directory to save network files')
cmd:option('--checkpoint',5,'save checkpoints')
cmd:option('--momentum',0.69,'initial momentum for training')
cmd:option('--clip',5,'max allowed norm ')
cmd:option('--XX',10,'the key parameter to determine the size of image, max is 39')
cmd:option('--RAM',false,'false means load all images to RAM')
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--imageType',1,'1: grayscale, 3: RGB')
cmd:text()
opt = cmd:parse(arg or {})

-- set up gpu
cutorch.setDevice(opt.gpu)
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true

XX=opt.XX

-- 1. Get the list of files in the given directory
--[[
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

files_lab ={}
for file in paths.files(opt.labelDir) do 
   if file:find(opt.ext .. '$') then
      table.insert(files_lab, paths.concat(opt.labelDir,file))
   end
end

if #files_lab == 0 then
   error('given directory doesnt contain any files of type: ' .. opt.ext)
end

table.sort(files_lab, function (a,b) return a < b end)

if not opt.RAM then
-- 2. Load all the files into RAM
-- "images" is a table of tensors of size opt.imageType x W x H 
   images = {}
   for i,file in ipairs(files) do
      -- load each image
      table.insert(images, image.load(file):float())
   end

   labels = {}
   for i, file in ipairs(files_lab) do 
      table.insert(labels, torch.reshape(image.load(file,1,'byte'),(16*XX-92)*(16*XX-92),1))
      --table.insert(labels, loader:forward(image.load(file,1,'byte')) )
   end
end
--]]

-- random data for test 

images={}
labels={}
files={'1','1','1','11','11'}

for kk=1,5 do
   table.insert(images,torch.rand(4, 1, 16*XX+92,16*XX+92):float())
   table.insert(labels,torch.ByteTensor(4, (16*XX-92),(16*XX-92)):random(1,2))
end



-- 3. Define the model 
--[[
if(images[1]:size(3)~=(16*XX+92)) then
   print('dimenstion mismatch')
   return
end
--]]

input = nn.Identity()()

L1a=cudnn.SpatialConvolution(opt.imageType, 64, 3, 3, 1, 1, 0, 0)(input)
L1b=cudnn.ReLU(true)(L1a)
L1c=cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0)(L1b)
if opt.dropoutProb>0 then
   L1d=nn.SpatialDropout(opt.dropoutProb)(L1c)
   L1=cudnn.ReLU(true)(L1d)
else
   L1=cudnn.ReLU(true)(L1c)
end


L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(L1)
L2b=cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0)(L2a)
L2c=cudnn.ReLU()(L2b)
L2d=cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0)(L2c)
if opt.dropoutProb>0 then
   L2e=nn.SpatialDropout(opt.dropoutProb)(L2d)
   L2=cudnn.ReLU(true)(L2e)
else
   L2=cudnn.ReLU(true)(L2d)
end

L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(L2)
L3b=cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0)(L3a)
L3c=cudnn.ReLU(true)(L3b)
L3d=cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0)(L3c)
if opt.dropoutProb>0 then
   L3e=nn.SpatialDropout(opt.dropoutProb)(L3d)
   L3=cudnn.ReLU(true)(L3e)
else
   L3=cudnn.ReLU(true)(L3d)
end

L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(L3)
L4b=cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0)(L4a)
L4c=cudnn.ReLU(true)(L4b)
L4d=cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0)(L4c)
if opt.dropoutProb>0 then
   L4e=nn.SpatialDropout(opt.dropoutProb)(L4d)
   L4=cudnn.ReLU(true)(L4e)
else
   L4=cudnn.ReLU(true)(L4d)
end

L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(L4)
L5b=cudnn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0)(L5a)
L5c=cudnn.ReLU(true)(L5b)
L5d=cudnn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0)(L5c)
if opt.dropoutProb>0 then
   L5e=nn.SpatialDropout(opt.dropoutProb)(L5d)
   L5=cudnn.ReLU(true)(L5e)
else
   L5=cudnn.ReLU(true)(L5d)
end

Crop4=nn.Narrow(2,4,2*XX-4)(L4)
L4cp=nn.Narrow(3,4,2*XX-4)(Crop4)
L5up=nn.SpatialFullConvolution(1024, 512, 2, 2, 2, 2)(L5)

L6a=nn.JoinTable(1,3)({L5up,L4cp})
L6b=cudnn.SpatialConvolution(1024,512, 3, 3, 1, 1, 0, 0)(L6a)
L6c=cudnn.ReLU(true)(L6b)
L6d=cudnn.SpatialConvolution(512,512, 3, 3, 1, 1, 0, 0)(L6c)
if opt.dropoutProb>0 then
   L6e=nn.SpatialDropout(opt.dropoutProb)(L6d)
   L6=cudnn.ReLU(true)(L6e)
else
   L6=cudnn.ReLU(true)(L6d)
end

Crop3=nn.Narrow(2,16,4*XX-16)(L3)
L3cp=nn.Narrow(3,16,4*XX-16)(Crop3)
L6up=nn.SpatialFullConvolution(512, 256, 2, 2, 2, 2)(L6)

L7a=nn.JoinTable(1,3)({L6up,L3cp})
L7b=cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 0, 0)(L7a)
L7c=cudnn.ReLU(true)(L7b)
L7d=cudnn.SpatialConvolution(256,256, 3, 3, 1, 1, 0, 0)(L7c)
if opt.dropoutProb>0 then
   L7e=nn.SpatialDropout(opt.dropoutProb)(L7d)
   L7=cudnn.ReLU(true)(L7e)
else
   L7=cudnn.ReLU(true)(L7d)
end

Crop2=nn.Narrow(2,40,8*XX-40)(L2)
L2cp=nn.Narrow(3,40,8*XX-40)(Crop2)
L7up=nn.SpatialFullConvolution(256, 128, 2, 2, 2, 2)(L7)

L8a=nn.JoinTable(1,3)({L7up,L2cp})
L8b=cudnn.SpatialConvolution(256,128, 3, 3, 1, 1, 0, 0)(L8a)
L8c=cudnn.ReLU(true)(L8b)
L8d=cudnn.SpatialConvolution(128,128, 3, 3, 1, 1, 0, 0)(L8c)
if opt.dropoutProb>0 then
   L8e=nn.SpatialDropout(opt.dropoutProb)(L8d)
   L8=cudnn.ReLU(true)(L8e)
else
   L8=cudnn.ReLU(true)(L8d)
end

Crop1=nn.Narrow(2,88,16*XX-88)(L1)
L1cp=nn.Narrow(3,88,16*XX-88)(Crop1)
L8up=nn.SpatialFullConvolution(128, 64, 2, 2, 2, 2)(L8)

L9a=nn.JoinTable(1,3)({L8up,L1cp})
L9b=cudnn.SpatialConvolution(128,64, 3, 3, 1, 1, 0, 0)(L9a)
L9c=cudnn.ReLU(true)(L9b)
L9d=cudnn.SpatialConvolution(64,64, 3, 3, 1, 1, 0, 0)(L9c)
if opt.dropoutProb>0 then
   L9e=nn.SpatialDropout(opt.dropoutProb)(L9d)
   L9=cudnn.ReLU(true)(L9e)
else
   L9=cudnn.ReLU(true)(L9d)
end

L10=cudnn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(L9)

--L10a=nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(L9)
--L10b=nn.Transpose({1,2},{2,3})(L10a)
--L10=nn.Reshape((16*XX-92)*(16*XX-92),2)(L10b)

unet = nn.gModule({input},{L10}):cuda()
--cudnn.convert(unet,cudnn):cuda()

local finput, fgradInput
unet:apply(function(m)  if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'nn.SpatialFullConvolution' then 
                           finput = finput or m.finput
                           fgradInput = fgradInput or m.fgradInput
                           m.finput = finput
                           m.fgradInput = fgradInput
                        end
            end)

--criterion = nn.CrossEntropyCriterion():cuda()
criterion = cudnn.SpatialCrossEntropyCriterion():cuda()

collectgarbage()

-- Training 
if opt.uniform > 0 then
   for k,param in ipairs(unet:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

parameters,gradParameters = unet:getParameters()

config = {learningRate=opt.learningRate,
          momentum=opt.momentum}
--config={learningRate=opt.learningRate, alpha=0.95}

function train()
   unet:training()
   epoch = epoch or 1

   if epoch%opt.checkpoint==0 then
      if config.learningRate > opt.minLR then
         config.learningRate = config.learningRate * 0.5
      end

      if config.momentum < 0.9 then
         config.momentum = config.momentum + 0.1
      end
   end 

   image_index = torch.randperm(#files):long()
   for i =1,#files do
      
      local feval = function (x)
         if x ~= parameters then parameters:copy(x) end
         gradParameters:zero()

         local idx = image_index[i]
         local input_image = images[idx]:cuda()
         local label_image = labels[idx]:cuda()
         --[[
         local input_image={}
         local label_image={}
         if not opt.RAM then
            table.insert(input_image,images[idx]:cuda())
            table.insert(label_image,labels[idx]:cuda())
         else
            table.insert(input_image, image.load(files[idx]):float():cuda())
            table.insert(label_image, image.load(files_lab[idx],1,'byte'):cuda())
         end
         --]]

         local output_image = unet:forward(input_image)
         local err = criterion:forward(output_image, label_image)
         local grad_df = criterion:backward(output_image, label_image)

         print('Epoch '..epoch..' ('..i..'): Err='..err)

         unet:backward(input_image,grad_df)

         if opt.clip>0 then
            gradParameters:clamp(-opt.clip, opt.clip)
         end

         return err, gradParameters
      end

      --optim.rmsprop(feval, parameters, config)
      optim.sgd(feval, parameters, config)

      -- clean 
      collectgarbage()
   end

   if opt.checkpoint>0  and epoch%opt.checkpoint ==0 then
      filename=string.format('%s/net_%f.bin',opt.CheckPointDir,epoch);
      torch.save(filename,unet);
   end

   epoch = epoch + 1
end

for iter=1, opt.epoch do
   train()
end


-- Traininig by Manual Loop 
--[[
lr = opt.learningRate
for k=1, opt.epoch do
   image_index = torch.randperm(#images):long()
   for i =1, #images do
      local idx = image_index[i]
      input_image=images[idx]:cuda()
      label_image=labels[idx]:cuda()

      output_image = unet:forward(input_image)
      local err = criterion:forward(output_image, label_image)
      local gradCriterion = criterion:backward(output_image, label_image)
      unet:zeroGradParameters()
      unet:backward(input_image,gradCriterion)
      unet:updateParameters(lr)

      print(label_image:float())

      print('Iter: '..k..' ('..i..'), Loss= '..err)

      if i%5==0 then
         collectgarbage()
      end
   end

   if (k % opt.checkpoint ==0) then
      filename=string.format('%s/net_%f.bin',opt.CheckPointDir,k);
      torch.save(filename,unet);
   end
end
--]]


