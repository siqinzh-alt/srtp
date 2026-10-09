function generate_multiclass_dataset(datasetName, nTrainPerClass, nValPerClass)
%GENERATE_MULTICLASS_DATASET Build a fast 5-class LLaVA dataset with a 7:3 split.
%   Default: 700 train and 300 validation examples per class (5,000 total).
%   Classes: BPSK, CW, LFM, FSK, and SHIP_NOISE.
%   Only spectrogram PNGs and LLaVA JSONL annotations are produced. Bellhop
%   propagation, WAV files, manifests, and metadata JSON are omitted because
%   LLaVA fine-tuning does not consume them.

if nargin < 1 || isempty(datasetName), datasetName = 'underwater_5class_v1'; end
if nargin < 2 || isempty(nTrainPerClass), nTrainPerClass = 700; end
if nargin < 3 || isempty(nValPerClass), nValPerClass = 300; end
assert(nTrainPerClass * 3 == nValPerClass * 7, ...
    '训练集和验证集每类数量必须满足 7:3。');

root = fileparts(fileparts(mfilename('fullpath')));
toolDir = fullfile(root, 'bellhop_tools');
addpath(toolDir, fileparts(mfilename('fullpath')), '-begin');
outDir = fullfile(root, 'output', 'datasets', datasetName);
assert(~exist(outDir, 'dir'), ...
    '输出目录已存在，为避免覆盖请使用新的 datasetName：%s', outDir);

noiseDirs.train = fullfile(root, '环境噪声', 'train');
noiseDirs.val = fullfile(root, '环境噪声', 'val');
classes = {'BPSK', 'CW', 'LFM', 'FSK', 'SHIP_NOISE'};
splits = {'train', 'val'};
counts = [nTrainPerClass, nValPerClass];

rng(20261007, 'twister');
mkdir(outDir);
cleanupOutput = onCleanup(@() removeIncompleteOutput(outDir));
sampleNo = 0;

for p = 1:numel(splits)
    split = splits{p};
    qaFile = fullfile(outDir, [split '_qa.jsonl']);
    prepareFile(qaFile);

    % 打乱类别生成顺序，避免 JSONL 和样本编号前段集中为单一类别；
    % 每个类别仍精确生成 counts(p) 条，因而不会改变 7:3 的类别配比。
    classSchedule = repelem(classes, counts(p));
    classSchedule = classSchedule(randperm(numel(classSchedule)));
    classProgress = zeros(size(classes));
    for k = 1:numel(classSchedule)
        className = classSchedule{k};
        classIndex = find(strcmp(classes, className), 1);
        classProgress(classIndex) = classProgress(classIndex) + 1;
            sampleNo = sampleNo + 1;
            id = sprintf('sample_%06d', sampleNo);
            if strcmp(className, 'SHIP_NOISE')
                s = struct('id', id, 'class_label', className, 'fs', 16000, ...
                    'dur', 3, 'image_px', 448);
                [rx, s.source] = make_ship_noise(noiseDirs.(split), s.fs, s.dur);
                s.receiver = struct('noise_type', 'ship_recording_as_target');
                s.channel = struct('simulator', 'none');
            else
                [rx, s] = makeFastSignalSample(className, id, noiseDirs.(split), split);
            end
            s.split = split;
            meta = saveLlavaOutput(rx, s, outDir, datasetName);
            appendJson(qaFile, make_multiclass_qa(meta));
            fprintf('%s %-10s [%d/%d], total [%d/%d]\n', split, className, ...
                classProgress(classIndex), counts(p), k, numel(classSchedule));
    end
end

writeReadme(outDir, datasetName, nTrainPerClass, nValPerClass, classes);
clear cleanupOutput; % The dataset completed successfully.
end

function [rx, s] = makeFastSignalSample(className, id, noiseDir, split)
% Use the source waveform plus real ambient noise; omit costly propagation.
s = make_multiclass_signal(className);
s.id = id;
s.class_label = className;
s.noise_dir = noiseDir;
s.noise_split = split;
[rx, s.receiver] = add_noise(s.tx, s);
s.channel = struct('simulator', 'omitted_for_fast_llava_generation');
end

function meta = saveLlavaOutput(rx, s, outDir, datasetName)
splitDir = fullfile(outDir, s.split);
imageDir = fullfile(splitDir, 'spectrogram');
if ~exist(imageDir, 'dir'), mkdir(imageDir); end
imagePath = fullfile(imageDir, [s.id '.png']);
saveSpectrogram(rx, s.fs, imagePath, s.image_px);

meta = struct('id', s.id, 'class_label', s.class_label, ...
    'spectrogram_path', datasetPath(datasetName, s.split, 'spectrogram', [s.id '.png']));
end

function qa = make_multiclass_qa(meta)
questions = {
    '请识别该水声时频图的信号类别。仅输出类别名称。\n<image>', ...
    '观察这张水声信号频谱图，它属于哪一类？只回答标准类别标签。\n<image>', ...
    '请判断图中的水声目标是 BPSK、CW、LFM、FSK 还是 SHIP_NOISE。仅输出一个类别名称。\n<image>', ...
    '根据时频纹理完成水声信号分类；不要解释，只给出类别。\n<image>', ...
    '这幅声学时频图对应的调制信号或噪声类型是什么？仅输出类别标签。\n<image>', ...
    '请对该水下声学图像进行五分类识别，答案限 BPSK、CW、LFM、FSK、SHIP_NOISE 之一。\n<image>', ...
    '分析图中频率随时间的变化特征，并给出信号类别名称；不要附加说明。\n<image>', ...
    '该样本最可能是何种水声通信信号或船舶噪声？仅返回类别名。\n<image>', ...
    '从这张时频图中辨认目标类别。请直接输出标签，不要描述图像。\n<image>', ...
    '请给出该水声片段的类别判定，输出格式为单个类别名称。\n<image>'};
questions = [questions, {
    '请阅读此水声时频表示，并判定其所属类别。答案只保留标签。\n<image>', ...
    '这段水下声学记录的可视化结果显示了哪种目标？仅输出标准分类名。\n<image>', ...
    '在给定的五个候选类别中，为图像选择最符合的一项。不要解释。\n<image>', ...
    '请根据能量分布和频率轨迹识别水声类型，只回复最终标签。\n<image>', ...
    '判别该谱图所表示的是调制通信信号还是船舶背景声，并输出具体类别。\n<image>', ...
    '请完成这一个水声时频样本的标签预测。输出限一个类别名称。\n<image>', ...
    '观察该图的窄带、跳频或扫频等特征，给出正确类别；不需要推理过程。\n<image>', ...
    '请从 BPSK、CW、LFM、FSK、SHIP_NOISE 中选出该图的真实标签。\n<image>', ...
    '作为水声信号识别任务，请对这张时频图做出单标签分类。\n<image>', ...
    '该声学片段在时频域中最符合哪一类别？请只输出类别。\n<image>', ...
    '请识别该水下目标的信号形态，回答一个规范类别名即可。\n<image>', ...
    '结合图像中的持续时间与频率结构，判断水声样本类别。仅给标签。\n<image>', ...
    '请为这张声谱图标注 BPSK、CW、LFM、FSK 或 SHIP_NOISE，不要附加文本。\n<image>', ...
    '请判定该图对应的波形/噪声类型。回复必须是唯一的类别名称。\n<image>', ...
    '分析该时频图并执行封闭集水声分类，直接给出分类结果。\n<image>', ...
    '请用一个标签概括该水声记录的类别，禁止给出解释。\n<image>', ...
    '这幅频谱图展示的目标属于哪个预定义类别？仅输出答案标签。\n<image>', ...
    '请对该水下声学样本进行信号类型辨识，输出单一类别。\n<image>', ...
    '依据可见的时频模式选择正确水声类别；仅回答类别名称。\n<image>', ...
    '识别此声学时频图中的信号类别。请不要输出分析过程。\n<image>'}];
questionIndex = mod(sum(double(meta.id)), numel(questions)) + 1;
conversations = struct('from', {'human', 'gpt'}, ...
    'value', {questions{questionIndex}, ...
              meta.class_label});
qa = struct('id', meta.id, 'image', meta.spectrogram_path, ...
    'conversations', conversations);
end

function saveSpectrogram(x, fs, path, px)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 px px]);
spectrogram(x, hann(512, 'periodic'), 448, 1024, fs, 'yaxis');
ax = gca;
ylim(ax, [0 fs / 2] / 1000);
axis(ax, 'off');
set(ax, 'Units', 'normalized', 'Position', [0 0 1 1]);
exportgraphics(ax, path, 'Resolution', 96);
close(fig);
end

function path = datasetPath(datasetName, split, folder, file)
path = strrep(fullfile('output', 'datasets', datasetName, split, folder, file), '\', '/');
end

function prepareFile(path)
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

function writeReadme(outDir, datasetName, nTrain, nVal, classes)
id = fopen(fullfile(outDir, 'README.md'), 'w');
assert(id ~= -1, 'Cannot write dataset README.');
closer = onCleanup(@() fclose(id));
fprintf(id, '# %s\n\n', datasetName);
fprintf(id, 'Five classes: %s.\n\n', strjoin(classes, ', '));
fprintf(id, 'Each class contains %d training and %d validation samples (7:3).\n\n', nTrain, nVal);
fprintf(id, '`train_qa.jsonl` and `val_qa.jsonl` use the LLaVA conversation format. ');
fprintf(id, 'Only spectrogram PNGs and LLaVA annotations are retained for fast training-data generation.\n');
end

function removeIncompleteOutput(outDir)
marker = fullfile(outDir, 'README.md');
if exist(outDir, 'dir') && ~exist(marker, 'file')
    rmdir(outDir, 's');
end
end
