require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('An image packing tool for DIV2K dataset')
cmd:text()
cmd:text('Options:')
cmd:option('-apath',        '../../../dataset',     'Absolute path of the DIV2K folder')
cmd:option('-dataset',      'DIV2K',                'Dataset to convert: DIV2K | Flickr2K')
cmd:option('-scale',        '2_3_4',                'Scales to pack')
cmd:option('-split',        'true',                 'split or pack')
cmd:option('-hr',           'true',                 'Generate HR data')
cmd:option('-lr',           'true',                 'Generate LR data')
cmd:option('-lrAug',        'true',                 'Generate pre-augmented unknown LR data')
cmd:option('-printEvery',   100,                    'print the progress # every iterations')

local opt = cmd:parse(arg or {})
opt.scale = opt.scale:split('_')
opt.split = opt.split == 'true'
opt.hr = opt.hr == 'true'
opt.lr = opt.lr == 'true'
opt.lrAug = opt.lrAug == 'true'
for i = 1, #opt.scale do
    opt.scale[i] = tonumber(opt.scale[i])
end

local targetPath, outputPath
local hrDir, lrDir

local targetPath = paths.concat(opt.apath, opt.dataset)
local outputPath = paths.concat(opt.apath, opt.dataset .. '_decoded')

lrDir = {}
if opt.lr then
    lrDir =  
    {
        'DIV2K_train_LR_bicubic_75',
    }
end

if not paths.dirp(outputPath) then
    paths.mkdir(outputPath)
end

local convertTable = {}

for i = 1, #lrDir do
    for j = 1, #opt.scale do
        local targetDir = paths.concat(targetPath, lrDir[i], 'X' .. opt.scale[j])
        local outputDir = paths.concat(outputPath, lrDir[i], 'X' .. opt.scale[j])
        if paths.dirp(targetDir) then
            if not paths.dirp(outputDir) then
                paths.mkdir(outputDir)
            end
            table.insert(convertTable, {tDir = targetDir, oDir = outputDir})
        end
    end
end

local ext = '.jpeg'
for i = 1, #convertTable do
    print('Converting ' .. convertTable[i].tDir)
    
    local imgTable = {}
    local n = 0
    local fileList = paths.dir(convertTable[i].tDir)
    table.sort(fileList)
    for j = 1, #fileList do
        if fileList[j]:find(ext) then
            local fileDir = paths.concat(convertTable[i].tDir, fileList[j])
            local img = image.load(fileDir, 3, 'byte')
            
            if opt.split then
                local fileName = fileList[j]:split(ext)[1] .. '.t7'
                torch.save(paths.concat(convertTable[i].oDir, fileName), img)
            else
                table.insert(imgTable, img)
            end

            n = n + 1
            if ((n % opt.printEvery) == 0) then
                print('Converted ' .. n .. ' files')
            end
        end
    end

    if not opt.split then
        torch.save(paths.concat(convertTable[i].oDir, 'pack.t7'), imgTable)
    end

    imageTable = nil
    collectgarbage()
    collectgarbage()
end
