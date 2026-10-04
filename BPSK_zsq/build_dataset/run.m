function run()

root = fileparts(fileparts(mfilename('fullpath')));% dataset 根目录
toolDir = fullfile(root, 'bellhop_tools');
addpath(toolDir, fileparts(mfilename('fullpath')), '-begin');

name = 'bpsk_v1';
outDir = fullfile(root, 'output', 'datasets', name);
noiseTrainDir = fullfile(root, '环境噪声', 'train');
noiseValDir = fullfile(root, '环境噪声','val');
 
px = 448;%%时频图尺寸待查找

nTrain = 10;    
nVal = 2;   

splits = {'train', 'val'};
counts = [nTrain, nVal];
sampleNo = 0;

for p = 1:numel(splits)%训练集和验证集
    split = splits{p};
    dataDir = fullfile(outDir, split);
    noiseDir = noiseTrainDir;
    if strcmp(split, 'val'), noiseDir = noiseValDir; end %切换验证集

    className = 'BPSK';

    % manifest = fullfile(outDir, [split '_manifest.jsonl']);%全部样本元信息清单
    qaFile = fullfile(outDir, [split '_qa.jsonl']);%多模态问答标注

    %prepareFile(manifest);
    prepareFile(qaFile);

        for k = 1:counts(p)
            sampleNo = sampleNo + 1;
            id = sprintf('sample_%06d', sampleNo);

            s = make_bpsk(1);%生成 BPSK 信号参数
            
            s.id = id;
            s.class_label = className;
            s.group_id = id;               % Future QA variants must keep this group ID.
            s.split = split;
            s.image_px = px;
            s.noise_dir = noiseDir;
            s.noise_split = split;

            channelDir = fullfile(dataDir, 'channel');
            [clean, channel] = add_bellhop_channel(s, channelDir);%加信道
            [rx, noiseInfo] = add_noise(clean, s);%加噪声
            meta = save_output(rx, s, channel, noiseInfo, dataDir, fullfile(name, split));%创建目录并输出文件
            
            % appendJson(manifest, meta);
            appendJson(qaFile, makeQa(meta));
            fprintf('%s %s [%d/%d]\n', split, className, k, counts(p));% 打印台进度
        end
end
end


function prepareFile(path)
folder = fileparts(path);
if ~exist(folder, 'dir'), mkdir(folder); end
id = fopen(path, 'w');%以写入模式打开文件：存在则清空内容，不存在则新建。返回文件标识符 id
assert(id ~= -1, 'Cannot write %s.', path);
fclose(id);
end

function appendJson(path, value)
id = fopen(path, 'a');
assert(id ~= -1, 'Cannot append to %s.', path);
closer = onCleanup(@() fclose(id));
fprintf(id, '%s\n', jsonencode(value));
end
