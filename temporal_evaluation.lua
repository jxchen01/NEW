require 'nn'
require 'rnn'
require 'ConvLSTM'
require 'optim'
require 'cutorch'
require 'cunn'
require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--dataDir', '/home/jchen16/NEW/data/fungus/encoder', 'the directory to load')
cmd:option('--ext','.t7','only load a specific type of files')
cmd:option('--rho',3,'maximum length of the sequence for each training iteration')
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--RAM',false,'true means load all images to RAM')
cmd:option('--OutputDir','/home/jchen16/NEW/code/test','the directoty to save checkpoints')
cmd:option('--HiddenSize','{32,16}','size of hidden layers')
cmd:option('--XX',20,'XX')
cmd:text()
opt = cmd:parse(arg or {})

opt.HiddenSize = dp.returnString(opt.HiddenSize)
XX=opt.XX
-- set GPU device
cutorch.setDevice(opt.gpu)

-------------------------------------------------------------------------------
---  Part1: Data Loading 
-------------------------------------------------------------------------------

-- load the result from encoder 
files = {}
for file in paths.files(opt.dataDir) do
   if file:find(opt.ext .. '$') then
      table.insert(files, paths.concat(opt.dataDir,file))
   end
end

if #files == 0 then
	error('given directory doesnt contain any files of type: ' .. opt.ext)
end
table.sort(files, function (a,b) return a < b end)

data = {}  -- "data" should be a table of data structures with field ('input','target')  
if opt.RAM then
	-- load all data
   	for i,file in ipairs(files) do
      	-- load each image
    	table.insert(data, torch.load(file))
   	end
else
	-- load one data to determine the size of data, which is necessary to define the model
	table.insert(data, torch.load(files[1])) 
end

print('finish loading data')

--[[
-- Test with randomly generated data 
files = {}
data = {}

for i=1,10 do 
	table.insert(files,'d')
	local inputs = {}
	local targets = {}
	for j=1, opt.rho do
		table.insert(inputs,torch.rand(64,16*XX-92,16*XX-92))
		table.insert(targets, torch.Tensor((16*XX-92)*(16*XX-92),1):random(1,2))
	end

	local inits = {}
	for j=1, #opt.HiddenSize do
		table.insert(inits, torch.Tensor(opt.HiddenSize[j],(16*XX-92),(16*XX-92)):random(1,2))
	end

	obj={input=inputs, target=targets,init=inits}
    table.insert(data,obj)
end
--]]

-------------------------------------------------------------------------------
---  Part2: Load the Model
-------------------------------------------------------------------------------
temporal_model=torch.load(opt.modelPath)
temporal_model:evaluate()

print('finish loading the model')

-------------------------------------------------------------------------------
---  Part3: Apply on each sequence 
-------------------------------------------------------------------------------

softmax = nn.SoftMax()
reshape_back = nn.Reshape((16*XX-92),(16*XX-92),4)

for i=1,2 do
--for i=1,#files do
	-- fetch one whole sequence 
	local input_sequence
	if not opt.RAW then
		a=torch.load(files[data_index[seq_idx]])
		input_sequence = a.input
	else
		input_sequence = data[data_index[seq_idx]].input
	end

		-- reset rnn memory
	for j=1, #opt.HiddenSize do
	  	temporal_model.module.module.modules[j]:forget()
	end

	for j=1, #input_sequence do
		input_sequence[j]=input_sequence[j]:cuda()
	end

		-- build initial cell state 
		-- for j=1, #opt.HiddenSize do
		--	init_cell_state = torch.Tensor(opt.HiddenSize[j],(16*XX-92),(16*XX-92)):copy(init_sequence[offset+1]:expand(opt.HiddenSize[j],(16*XX-92),(16*XX-92)))
	 	--	temporal_model.module.module.modules[j].userPrevCell = init_cell_state:cuda()
		-- end
	local output_sequence = temporal_model:forward(input_sequence)
	for j=1, #output_sequence do
		local filename=string.format('%s/v1_%d_%d.png',opt.OutputDir,i,j);
		local c = softmax:forward(output_sequence[j]:double())
		local d = reshape_back:forward(c)
		local img = d:select(3,4)
		image.save(filename, img)
	end

    -- clean 
    collectgarbage()
end




