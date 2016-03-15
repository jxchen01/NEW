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
cmd:option('--outputPath', '/home/jchen16/NEW/data/temporal/output', 'the directory to save outputs')
cmd:option('--modelPath','/home/jchen16/NEW/checkpoint/brnn_16500.000000.bin', 'the directory to the model')
cmd:option('--ext','.t7','only load a specific type of files')
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--RAM',false,'true means load all images to RAM')
cmd:option('--HiddenSize',{128,64},'size of hidden layers')
cmd:option('--XX',10,'XX')
cmd:option('--bi',false,'use bi-directional model or not')
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
---  Part2: Load trained Model
-------------------------------------------------------------------------------

temporal_model = torch.load(opt.modelPath)
temporal_model:evaluate()

--[[
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
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for j=1, #opt.HiddenSize do
	temporal_model.module.module.modules[j]:initBias(1,0)
end

print('finish building the model')
--]]

-------------------------------------------------------------------------------
---  Part3: Testing
-------------------------------------------------------------------------------
softmax = nn.SoftMax()
reshape_back = nn.Reshape((16*XX-92),(16*XX-92),2)

for i=1, #files do

	if not opt.RAW then
		a=torch.load(files[i])
		input_sequence = a.input
		init = a.init[1]
	else
		input_sequence = data[i].input
		init = data[i].init[1]
	end
		
	-- reset rnn memory
	if opt.bi then
		temporal_model.modules[1].backwardModule:forget()
	else
		for j=1, #opt.HiddenSize do
	  		temporal_model.module.module.modules[j]:forget()
		end
	end

	-- build initial cell state 
	if not opt.bi then
		for j=1, #opt.HiddenSize do
			init_cell_state = torch.Tensor(opt.HiddenSize[j],(16*XX-92),(16*XX-92)):copy(init:expand(opt.HiddenSize[j],(16*XX-92),(16*XX-92)))
	 		temporal_model.module.module.modules[j].userPrevCell = init_cell_state:cuda()
		end
	end

	local output_sequence=temporal_model:forward(input_sequence)

	for j=1, #output_sequence do
		local c=softmax:forward(output_sequence[j])
		local d=reshape_back:forward(c)
		local ff=d:select(3,2)
		local str= string.format('%s/test_%d_%d.png',opt.outputPath,i,j);
		image.save(str, ff)
	end
		
    -- clean 
    collectgarbage()
end

