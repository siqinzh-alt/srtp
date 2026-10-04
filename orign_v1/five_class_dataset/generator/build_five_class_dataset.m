function build_five_class_dataset()

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
toolDir = fullfile(root, 'bpsk_dataset_clean', 'bellhop_tools');
sourceGen = fullfile(root, 'bpsk_dataset_clean', 'dataset_generator');
addpath(toolDir, sourceGen, '-begin');

name = 'four_class_v1';
outDir = fullfile(root, 'five_class_dataset', 'output', 'datasets', name);
noiseRoot = fullfile(root, '环境噪声');
noiseTrainDir = fullfile(noiseRoot, 'train');
noiseValDir = fullfile(noiseRoot, 'val');
woaDir = fullfile(root, 'WOA');
fs = 16000;
dur = 5;
px = 448;
nTrain = 100;    % Samples per class. Use 10 first to check the pipeline.
nVal = 20;       % Samples per class. Keep class counts balanced.

assert(exist(noiseTrainDir, 'dir') == 7, ...
    '训练噪声文件夹不存在：%s', noiseTrainDir);
assert(exist(noiseValDir, 'dir') == 7, ...
    '验证噪声文件夹不存在：%s', noiseValDir);
assert(exist(woaDir, 'dir') == 7, 'WOA 文件夹不存在：%s', woaDir);
assert(~isempty(noiseWavs(noiseTrainDir)), ...
    '训练噪声文件夹中没有 WAV：%s', noiseTrainDir);
assert(~isempty(noiseWavs(noiseValDir)), ...
    '验证噪声文件夹中没有 WAV：%s', noiseValDir);

rng(20260929, 'twister');
classes = {'CW', 'LFM', 'FSK', 'BPSK'};
splits = {'train', 'val'};
counts = [nTrain, nVal];
sampleNo = 0;

for p = 1:numel(splits)
    split = splits{p};
    dataDir = fullfile(outDir, split);
    noiseDir = noiseTrainDir;
    if strcmp(split, 'val'), noiseDir = noiseValDir; end
    manifest = fullfile(outDir, [split '_manifest.jsonl']);
    qaFile = fullfile(outDir, [split '_qa.jsonl']);
    prepareFile(manifest);
    prepareFile(qaFile);
    for c = 1:numel(classes)
        className = classes{c};
        for k = 1:counts(p)
            sampleNo = sampleNo + 1;
            id = sprintf('sample_%06d', sampleNo);
            s = makePulseSample(id, className, split, fs, dur, px, ...
                noiseDir, woaDir);
            meta = generate_bellhop_sample(s, dataDir, fullfile(name, split));
            appendJson(manifest, meta);
            appendJson(qaFile, makeQa(meta));
            fprintf('%s %s [%d/%d]\n', split, className, k, counts(p));
        end
    end
end
end

function s = makePulseSample(id, className, split, fs, dur, px, noiseDir, woaDir)
[z, temp, woa] = pick_woa_temperature(woaDir, 100);
active = 2.0;
start = 1.0 + rand * (dur - active - 2.0);

switch className
    case 'CW'
        sig = struct('type', 'CW', 'carrier_hz', randi([1200 1800]), ...
            'start_time_s', start, 'pulse_width_s', active, 'fade_s', 0.02);
    case 'LFM'
        sig = struct('type', 'LFM', 'f_start_hz', randi([2200 2600]), ...
            'f_end_hz', randi([5200 6000]), 'start_time_s', start, ...
            'pulse_width_s', active, 'fade_s', 0.02);
    case 'FSK'
        sig = struct('type', '2FSK', 'tone_hz', [3800 4800], ...
            'symbol_rate_hz', randi([100 180]), 'start_time_s', start, ...
            'pulse_width_s', active, 'fade_s', 0.01);
    case 'BPSK'
        sig = struct('type', 'BPSK', 'carrier_hz', randi([3000 3400]), ...
            'chips_per_sec', randi([500 800]), 'start_time_s', start, ...
            'pulse_width_s', active, 'fade_s', 0.02);
    otherwise
        error('Unsupported class: %s', className);
end

depths = z(2:end-1);
s.id = id;
s.class_label = className;
s.group_id = id;               % Future QA variants must keep this group ID.
s.split = split;
s.sig = sig;
s.fs = fs;
s.dur = dur;
s.snr_db = randi([10 18]);     % Clear pilot set; reduce later for a harder set.
s.image_px = px;
s.z = z;
s.temp = temp;
s.woa = woa;
s.sd = depths(randi(numel(depths)));
s.rd = depths(randi(numel(depths)));
s.rr = 0.6 + 0.6 * rand;
s.noise_dir = noiseDir;
s.noise_split = split;
end

function qa = makeQa(meta)
qa = struct('id', meta.id, 'group_id', meta.group_id, 'split', meta.split, ...
    'image', meta.spectrogram_path, ...
    'question', ['请根据时频图判断水声信号大类。候选类别：CW、LFM、FSK、BPSK。' ...
    '仅输出一个类别名称。'], ...
    'answer', meta.class_label);
end

function files = noiseWavs(noiseDir)
files = dir(fullfile(noiseDir, '*.wav'));
files = files(~[files.isdir]);
end

function prepareFile(path)
folder = fileparts(path);
if ~exist(folder, 'dir'), mkdir(folder); end
id = fopen(path, 'w');
assert(id ~= -1, 'Cannot write %s.', path);
fclose(id);
end

function appendJson(path, value)
id = fopen(path, 'a');
assert(id ~= -1, 'Cannot append to %s.', path);
closer = onCleanup(@() fclose(id));
fprintf(id, '%s\n', jsonencode(value));
end
