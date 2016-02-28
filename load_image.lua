require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--dir', './data', 'the directory to load')
cmd:option('--ext','.png','only load a specific type of images')
cmd:text()
opt = cmd:parse(arg or {})

-- 1. Get the list of files in the given directory

files = {}

for file in paths.files(opt.dir) do
   	if file:find(opt.ext .. '$') then
      table.insert(files, paths.concat(opt.dir,file))
   	end
end

if #files == 0 then
	error('given directory doesnt contain any files of type: ' .. opt.ext)
end

table.sort(files, function (a,b) return a < b end)

print('Found files:')
print(files)

-- 2. Load all the files in the list

images = {}
for i,file in ipairs(files) do
   	-- load each image
   	table.insert(images, image.load(file))
end

print('Loaded images:')
print(images)

