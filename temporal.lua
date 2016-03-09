require 'nn'
require 'rnn'
require 'convLSTM'
require 'optim'
require 'cutorch'
require 'cunn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--featureMapDir', '/home/jchen16/NEW/data/X10/fm/', 'the directory to load')
cmd:option('--ext','.t7','only load a specific type of files')
cmd:option('--rho',3,'maximum length of the sequence for each training iteration')
cmd:option('--kernalSize',7,'the kernal size of convolution on input feature map')
cmd:option('--kernalSizeMemory',10,'the kernal size of convolution on the cell state')
cmd:option('--learningRate',0.05,'initial learning rate')
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--RAM',false,'false means load all images to RAM')
cmd:option('--clip',5,'max allowed gradient norm in BPTT')
cmd:option('--randNorm', 0.05, 'initialize parameters using uniform distribution between -uniform and uniform.')
cmd:option('--checkpoint',100,'the number of iteration to save checkpoints')
cmd:option('--nIteration',10,'the number of training iterations')
cmd:text()
opt = cmd:parse(arg or {})

XX=10

-- set GPU device
cutorch.setDevice(opt.gpu)

-------------------------------------------------------------------------------
---  Part1: Data Loading 
-------------------------------------------------------------------------------
--[[
-- load the result from encoder 
files = {}
for file in paths.files(opt.featureMapDir) do
   if file:find(opt.ext .. '$') then
      table.insert(files, paths.concat(opt.featureMapDir,file))
   end
end

if #files == 0 then
	error('given directory doesnt contain any files of type: ' .. opt.ext)
end
table.sort(files, function (a,b) return a < b end)

data = {}  -- "data" should be a table of data structures with field ('input','target','init')  
if not opt.RAM then
	-- load all data
   	for i,file in ipairs(files) do
      	-- load each image
    	table.insert(data, torch.load(file))
   	end
else
	-- load one data to determine the size of data, which is necessary to define the model
	table.insert(data, torch.load(files[1])) 
end
--]]

-- Test with randomly generated data 
files = {}
data = {}

for i=1,10 do 
	table.insert(files,'d')
	obj={input=torch.rand(opt.rho,64,16*XX-92,16*XX-92), 
		 target= torch.Tensor(opt.rho,(16*XX-92)*(16*XX-92),2):bernoulli(0.5),
		 init={torch.Tensor(2,16*XX-92,16*XX-92):bernoulli(0.5),torch.Tensor(2,16*XX-92,16*XX-92):bernoulli(0.5)}}
    table.insert{data,obj}
end

-------------------------------------------------------------------------------
---  Part2: Model and Criterion
-------------------------------------------------------------------------------

-- build the model 
inputDepth = data[1].input:size(2) -- the number of features (dimension: {seq, featre, w, h})
HiddenSize={128} -- {128,64}
local temporal_model = nn.Sequential()
for i, temporalSize in ipairs(HiddenSize) do
	--seq = nn.ConvLSTM(inputDepth,temporalSize, 3, 7, 9, 1)
	local seq = nn.ConvLSTM(inputDepth,temporalSize, 3, opt.kernelSize, opt.kernelSizeMemory, 1)
	seq:remember('both')
	seq:training()
	inputDepth = temporalSize
	temporal_model:add(seq)
end

temporal_model:add(nn.SpatialConvolution(inputDepth, 2, 1, 1, 1, 1, 0, 0))
temporal_model:add(nn.Transpose({1,2},{2,3}))
temporal_model:add(nn.Reshape(data[1].input:size(2)*data[1].input:size(2),2))
temporal_model = nn.Sequencer(temporal_model)  -- decorate with Sequencer()

-- ship the model to gpu
temporal_model:cuda()

-- define criterion and ship to gpu
criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion()):cuda()

-- parameters initialization
params, gradParams = temporal_model:getParameters()
if opt.randNorm>0 then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for j=1, #HiddenSize do
	temporal_model.module.module.modules[j]:initBias(1,0)
end
-------------------------------------------------------------------------------
---  Part3: Training 
-------------------------------------------------------------------------------
local data_index = torch.randperm(#files):long() -- feed the training sequences in a random order
local seq_idx=1
temporal_model:training()
local optim_config = {learningRate = opt.learning_rate}
for i=1, opt.nIteration do
	-- prepare a sequence of rho frames
	if torch.mod(seq_idx,#files)==0 then
		data_index = torch.randperm(#files):long()
		seq_idx=1;
	end
	inputs = data[data_index[seq_idx]].input:cuda()
	targets = data[data_index[seq_idx]].target:cuda()

	-- build initial cell state 
	init_state= data[data_index[seq_idx]].init:cuda()
	for j=1, #HiddenSize do
		temporal_model.module.module.modules[j].userPrevCell = init_state[j]
	end

	-- reset rnn memory
	for j=1, #HiddenSize do
		temporal_model.module.module.modules[j]:forget()
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

	local _, loss = optim.adam(feval, params, optim_config)

	-- clean 
    collectgarbage()

    if opt.checkpoint>0 and i%opt.checkpoint then
    	filename=string.format('%s/rnn_%f.bin',opt.CheckPointDir,i);
      	torch.save(filename,temporal_model);
   	end

    seq_idx = seq_idx + 1
end
