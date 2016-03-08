require 'nn'
require 'rnn'
require 'ConvLSTM'
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
cmd:option('--gpu',1,'gpu device to use')
cmd:option('--RAM',false,'false means load all images to RAM')
cmd:option('--randNorm', 0.05, 'initialize parameters using uniform distribution between -uniform and uniform.')
-- set GPU device
cutorch.setDevice(opt.gpu)

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

fm = {}  -- "fm" should be a table of tensors of size k x W x H 
if not opt.RAM then
	-- load all data
   	for i,file in ipairs(files) do
      	-- load each image
    	table.insert(fm, torch.load(file))
   	end
else
	-- load one data to determine the size of data, which is necessary to define the model
	table.insert(fm, torch.load(files[1])) 
end

-- build the model 
inputDepth = fm[1].size(1) -- the number of features (fm: feature map)
HiddenSize={128} -- {128,64}
local temporal_model = nn.Sequential()
for i, temporalSize in ipairs(HiddenSize) do
	local seq = nn.ConvLSTM(inputDepth,temporalSize, 3, opt.kernelSize, opt.kernelSizeMemory, 1)
	seq:remember('both')
	seq:training()
	inputDepth = temporalSize
	temporal_model:add(seq)
end
temporal_model:add(nn.SpatialConvolution(inputDepth, 2, 1, 1, 1, 1, 0, 0)
temporal_model:add(nn.Transpose({1,2},{2,3})
temporal_model:add(nn.Reshape(fm[1].size(2)*fm[1].size(3),2)
temporal_model = nn.Sequencer(temporal)  -- decorate with Sequencer()

-- ship the model to gpu
temporal_model:cuda()

-- build initial cell state 
init_state= {}
for i=1, #HiddenSize do

end

params, gradParams = temporal_model:getParameters()
-- parameters initialization
if opt.randNorm>0 then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning


-- training 
for i=1, opt.nIteration do
	-- sequence of rho frames
	
end


--[[
-- define the evaluation closure 
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    gradParams:zero()

    ------ forward -------
    for t=1,opt.nSeq do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end

end
--]]



