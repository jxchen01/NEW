require 'nn'
require 'rnn'
require 'ConvLSTM'
require 'optim'
require 'cutorch'
require 'cunn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--dataDir', '/home/jchen16/NEW/data/temporal/encoder', 'the directory to load')
cmd:option('--ext','.t7','only load a specific type of files')
cmd:option('--rho',3,'maximum length of the sequence for each training iteration')
cmd:option('--kernalSize',15,'the kernal size of convolution on input feature map')
cmd:option('--kernalSizeMemory',7,'the kernal size of convolution on the cell state')
cmd:option('--learningRate',0.05,'initial learning rate')
cmd:option('--minLR',0.0005,'minimal learning rate')
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--RAM',false,'true means load all images to RAM')
cmd:option('--clip',5,'max allowed gradient norm in BPTT')
cmd:option('--randNorm', 0.08, 'initialize parameters using uniform distribution between -uniform and uniform.')
cmd:option('--checkpoint',500,'the number of iteration to save checkpoints')
cmd:option('--CheckPointDir','/home/jchen16/NEW/code/checkpoint','the directoty to save checkpoints')
cmd:option('--nIteration',40000,'the number of training iterations')
cmd:option('--HiddenSize',{128},'size of hidden layers')
cmd:option('--XX',10,'XX')
cmd:text()
opt = cmd:parse(arg or {})

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

data = {}  -- "data" should be a table of data structures with field ('input','target','init')  
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
---  Part2: Model and Criterion
-------------------------------------------------------------------------------

-- build the model 
inputDepth = data[1].input[1]:size(1) -- the number of features (dimension: {featre, w, h})
local temporal_model = nn.Sequential()
for i, temporalSize in ipairs(opt.HiddenSize) do
	local seq = nn.ConvLSTM(inputDepth,temporalSize, opt.rho, opt.kernalSize, opt.kernalSizeMemory, 1)
	seq:remember('both')
	seq:training()
	inputDepth = temporalSize
	temporal_model:add(seq)
end
temporal_model:add(nn.SpatialConvolution(inputDepth, 2, 1, 1, 1, 1, 0, 0))
temporal_model:add(nn.Transpose({1,2},{2,3}))
temporal_model:add(nn.Reshape(data[1].input[1]:size(2)*data[1].input[1]:size(3),2))
temporal_model = nn.Sequencer(temporal_model)  -- decorate with Sequencer()

-- ship the model to gpu
temporal_model:cuda()

-- define criterion and ship to gpu
criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()):cuda()


-- parameters initialization
params, gradParams = temporal_model:getParameters()
if opt.randNorm>0 then
    params:uniform(-opt.randNorm, opt.randNorm) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for j=1, #opt.HiddenSize do
	temporal_model.module.module.modules[j]:initBias(0,0)
end

print('finish building the model')

-------------------------------------------------------------------------------
---  Part3: Training 
-------------------------------------------------------------------------------
local data_index = torch.randperm(#files):long() -- feed the training sequences in a random order
local seq_idx=1
--temporal_model:training()
local optim_config = {learningRate = opt.learningRate, momentum= 0.59}
for i=1, opt.nIteration do
	
	-- fetch one whole sequence 
	if seq_idx%(#files)==0 then
		if optim_config.learningRate > opt.minLR then
			optim_config.learningRate = optim_config.learningRate * 0.75
		end
		if optim_config.momentum <0.99 then
			optim_config.momentum = optim_config.momentum + 0.025
		end
		data_index = torch.randperm(#files):long()
		seq_idx=1;
	end

	if not opt.RAW then
		a=torch.load(files[data_index[seq_idx]])
		input_sequence = a.input
		target_sequence = a.target
		init_sequence = a.init
	else
		input_sequence = data[data_index[seq_idx]].input
		target_sequence = data[data_index[seq_idx]].target
		init_sequence = data[data_index[seq_idx]].init
	end

	-- prepare a sequence of rho frames
    for offset = 0, #input_sequence-opt.rho do
    	local inputs={}
		local targets={}
		for j=1,opt.rho do 
			table.insert(inputs,input_sequence[j+offset]:cuda())
			table.insert(targets,target_sequence[j+offset]:cuda())
		end

		-- reset rnn memory
		for j=1, #opt.HiddenSize do
	  		temporal_model.module.module.modules[j]:forget()
		end

		-- build initial cell state 
		for j=1, #opt.HiddenSize do
			init_cell_state = torch.Tensor(opt.HiddenSize[j],(16*XX-92),(16*XX-92)):copy(init_sequence[offset+1]:expand(opt.HiddenSize[j],(16*XX-92),(16*XX-92)))
	 		temporal_model.module.module.modules[j].userPrevCell = init_cell_state:cuda()
		end

		-- define the evaluation closure 
		local feval = function (x)
    		if x ~= params then params:copy(x) end
    		gradParams:zero()

    		------ forward -------
			local outputs=temporal_model:forward(inputs)
			local err = criterion:forward(outputs,targets)

			------ backward ------
			local gradOutputs = criterion:backward(outputs,targets)
			local gradInputs = temporal_model:backward(inputs, gradOutputs)

			if opt.clip>0 then
				gradParams:clamp(-opt.clip, opt.clip)
			end

			return err, gradParams
		end

		local _, loss = optim.sgd(feval, params, optim_config)
		print('Iter '..i..'('..offset..'), Loss = '..loss[1])
    end

    -- clean 
    collectgarbage()

    if opt.checkpoint>0 and i%opt.checkpoint==0 then
    	filename=string.format('%s/rnn_%f.bin',opt.CheckPointDir,i);
    	--temporal_model:clearState()
      	torch.save(filename,temporal_model);
   	end

    seq_idx = seq_idx + 1
end

