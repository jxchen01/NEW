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
cmd:option('--imageDir', '/home/jchen16/NEW/data/train/', 'the directory to load')
cmd:option('--labelDir', '/home/jchen16/NEW/data/label/', 'the directory to load')
cmd:option('--ext','.png','only load a specific type of images')
cmd:option('--epoch',5000,'the number of iterations trained on the whole dataset')
cmd:option('--learningRate',0.0001,'initial learning rate')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform.')
cmd:option('--CheckPointDir', './checkpoint','directory to save network files')
cmd:option('--checkpoint',false,'save checkpoints')
cmd:option('--momentum',0.99,'momentum for training')
cmd:option('--clip',2,'max allowed norm ')
cmd:text()
opt = cmd:parse(arg or {})

XX=10

-- to do:
-- nninit
-- dynamic lr 
-- maxNorm 
-- normCutOff

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

-- 2. Load all the files into RAM
-- "images" is a table of tensors of size 1 x L x L 
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

loader = nn.Reshape((16*XX-92)*(16*XX-92),1)

labels = {}
for i, file in ipairs(files_lab) do 
   table.insert(labels, loader:forward(image.load(file)) )
end

-- random data for test 
--[[
images={}
labels={}

for kk=1,5 do
   table.insert(images,torch.rand(1,16*XX+92,16*XX+92))
   table.insert(labels,torch.Tensor((16*XX-92)*(16*XX-92),1):random(1,2))
end
--]]


-- 3. Define the model 
if(images[1]:size(3)~=(16*XX+92)) then
   print('dimenstion mismatch')
   return
end

input = nn.Identity()()

L1a=nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 0, 0)(input)
L1b=nn.ReLU(true)(L1a)
L1c=nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0)(L1b)
if opt.dropoutProb>0 then
   L1d=nn.SpatialDropout(opt.dropoutProb)(L1c)
   L1=nn.ReLU(true)(L1d)
else
   L1=nn.ReLU(true)(L1c)
end


L2a=nn.SpatialMaxPooling(2, 2, 2, 2)(L1)
L2b=nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0)(L2a)
L2c=nn.ReLU()(L2b)
L2d=nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0)(L2c)
if opt.dropoutProb>0 then
   L2e=nn.SpatialDropout(opt.dropoutProb)(L2d)
   L2=nn.ReLU(true)(L2e)
else
   L2=nn.ReLU(true)(L2d)
end

L3a=nn.SpatialMaxPooling(2, 2, 2, 2)(L2)
L3b=nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 0, 0)(L3a)
L3c=nn.ReLU(true)(L3b)
L3d=nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 0, 0)(L3c)
if opt.dropoutProb>0 then
   L3e=nn.SpatialDropout(opt.dropoutProb)(L3d)
   L3=nn.ReLU(true)(L3e)
else
   L3=nn.ReLU(true)(L3d)
end

L4a=nn.SpatialMaxPooling(2, 2, 2, 2)(L3)
L4b=nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 0, 0)(L4a)
L4c=nn.ReLU(true)(L4b)
L4d=nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 0, 0)(L4c)
if opt.dropoutProb>0 then
   L4e=nn.SpatialDropout(opt.dropoutProb)(L4d)
   L4=nn.ReLU(true)(L4e)
else
   L4=nn.ReLU(true)(L4d)
end

L5a=nn.SpatialMaxPooling(2, 2, 2, 2)(L4)
L5b=nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 0, 0)(L5a)
L5c=nn.ReLU(true)(L5b)
L5d=nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 0, 0)(L5c)
if opt.dropoutProb>0 then
   L5e=nn.SpatialDropout(opt.dropoutProb)(L5d)
   L5=nn.ReLU(true)(L5e)
else
   L5=nn.ReLU(true)(L5d)
end

Crop4=nn.Narrow(2,4,2*XX-4)(L4)
L4cp=nn.Narrow(3,4,2*XX-4)(Crop4)
L5up=nn.SpatialFullConvolution(1024, 512, 2, 2, 2, 2)(L5)

L6a=nn.JoinTable(1,3)({L5up,L4cp})
L6b=nn.SpatialConvolution(1024,512, 3, 3, 1, 1, 0, 0)(L6a)
L6c=nn.ReLU(true)(L6b)
L6d=nn.SpatialConvolution(512,512, 3, 3, 1, 1, 0, 0)(L6c)
if opt.dropoutProb>0 then
   L6e=nn.SpatialDropout(opt.dropoutProb)(L6d)
   L6=nn.ReLU(true)(L6e)
else
   L6=nn.ReLU(true)(L6d)
end

Crop3=nn.Narrow(2,16,4*XX-16)(L3)
L3cp=nn.Narrow(3,16,4*XX-16)(Crop3)
L6up=nn.SpatialFullConvolution(512, 256, 2, 2, 2, 2)(L6)

L7a=nn.JoinTable(1,3)({L6up,L3cp})
L7b=nn.SpatialConvolution(512,256, 3, 3, 1, 1, 0, 0)(L7a)
L7c=nn.ReLU(true)(L7b)
L7d=nn.SpatialConvolution(256,256, 3, 3, 1, 1, 0, 0)(L7c)
if opt.dropoutProb>0 then
   L7e=nn.SpatialDropout(opt.dropoutProb)(L7d)
   L7=nn.ReLU(true)(L7e)
else
   L7=nn.ReLU(true)(L7d)
end

Crop2=nn.Narrow(2,40,8*XX-40)(L2)
L2cp=nn.Narrow(3,40,8*XX-40)(Crop2)
L7up=nn.SpatialFullConvolution(256, 128, 2, 2, 2, 2)(L7)

L8a=nn.JoinTable(1,3)({L7up,L2cp})
L8b=nn.SpatialConvolution(256,128, 3, 3, 1, 1, 0, 0)(L8a)
L8c=nn.ReLU(true)(L8b)
L8d=nn.SpatialConvolution(128,128, 3, 3, 1, 1, 0, 0)(L8c)
if opt.dropoutProb>0 then
   L8e=nn.SpatialDropout(opt.dropoutProb)(L8d)
   L8=nn.ReLU(true)(L8e)
else
   L8=nn.ReLU(true)(L8d)
end

Crop1=nn.Narrow(2,88,16*XX-88)(L1)
L1cp=nn.Narrow(3,88,16*XX-88)(Crop1)
L8up=nn.SpatialFullConvolution(128, 64, 2, 2, 2, 2)(L8)

L9a=nn.JoinTable(1,3)({L8up,L1cp})
L9b=nn.SpatialConvolution(128,64, 3, 3, 1, 1, 0, 0)(L9a)
L9c=nn.ReLU(true)(L9b)
L9d=nn.SpatialConvolution(64,64, 3, 3, 1, 1, 0, 0)(L9c)
if opt.dropoutProb>0 then
   L9e=nn.SpatialDropout(opt.dropoutProb)(L9d)
   L9=nn.ReLU(true)(L9e)
else
   L9=nn.ReLU(true)(L9d)
end

--L10=nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(L9)

L10a=nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(L9)
L10b=nn.Transpose({1,2},{2,3})(L10a)
L10=nn.Reshape((16*XX-92)*(16*XX-92),2)(L10b)

unet = nn.gModule({input},{L10}):cuda()

local finput, fgradInput
unet:apply(function(m) if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'nn.SpatialFullConvolution' then 
                           finput = finput or m.finput
                           fgradInput = fgradInput or m.fgradInput
                           m.finput = finput
                           m.fgradInput = fgradInput
                        end
            end)


criterion = nn.CrossEntropyCriterion():cuda()

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


function train()
   unet:training()
   epoch = epoch or 1

   image_index = torch.randperm(#images):long()
   for i =1,#images do
      
      local feval = function (x)
         if x ~= parameters then parameters:copy(x) end
         gradParameters:zero()

         local idx = image_index[i]
         local input_image=images[idx]:cuda()
         local label_image=labels[idx]:cuda()

         local output_image = unet:forward(input_image)
         local err = criterion:forward(output_image, label_image)
         local grad_df = criterion:backward(output_image, label_image)

         --print('Epoch '..epoch..' ('..i..'): Err='..err)

         unet:backward(input_image,grad_df)

         if opt.clip>0 then
            gradParameters:clamp(-opt.clip, opt.clip)
         end

         return err, gradParameters
      end

      tmp , {f} = optim.sgd(feval, parameters, config)
      print(f)
   end

   epoch = epoch + 1
end

for iter=1, opt.epoch do
   train()
   collectgarbage()
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


