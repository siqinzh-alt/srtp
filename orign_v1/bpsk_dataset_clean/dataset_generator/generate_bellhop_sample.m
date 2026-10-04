function meta = generate_bellhop_sample(s, dataRoot, datasetName)


audioDir = fullfile(dataRoot, 'audio');
specDir = fullfile(dataRoot, 'spectrogram');
metaDir = fullfile(dataRoot, 'metadata');
channelDir = fullfile(dataRoot, 'channel');
dirs = {audioDir, specDir, metaDir, channelDir};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir'), mkdir(dirs{k}); end
end

fc = signal_center_frequency(s.sig);

[SSP, Bdry, Pos, Beam, cInt, RMax, c] = ...
    build_ssp(s.z, s.temp, s.sd, s.rd, s.rr);%构建 SSP 剖面

env = fullfile(channelDir, [s.id '.env']);
arr = fullfile(channelDir, [s.id '.arr']);
write_env(env, 'BELLHOP', s.id, fc, SSP, Bdry, Pos, Beam, cInt, RMax);

old = pwd;%获取当前目录路径
restoreFolder = onCleanup(@() cd(old));
cd(channelDir);
bellhop(s.id);%调用 Bellhop 水声射线模型

[Arr, ~] = read_arrivals_local(arr, 500);

[tx, label, source] = generate_signal(s.sig, s.fs, s.dur);%待修改信号
[clean, delayS, amp] = delayandsum(tx, s.fs, Arr, 1, 1, 1);%返回各径时延 delayS 和复幅度 amp
noiseDir = getNoiseDir(s, dataRoot);
noiseFiles = getField(s, 'noise_files', []);
[rx, noiseInfo] = add_real_noise(clean, s.fs, s.snr_db, noiseDir, noiseFiles);
if isfield(s, 'noise_split'), noiseInfo.source_split = s.noise_split; end
if max(abs(rx)) > 0.999, rx = 0.999*rx/max(abs(rx)); end

wav = fullfile(audioDir, [s.id '.wav']);
png = fullfile(specDir, [s.id '.png']);
json = fullfile(metaDir, [s.id '.json']);
audiowrite(wav, rx, s.fs, 'BitsPerSample', 16);

saveSpectrogram(rx, s.fs, png, getField(s, 'image_px', 448));

meta.id = s.id;
meta.label = label;
meta.audio_path = relPath(datasetName, 'audio', [s.id '.wav']);
meta.spectrogram_path = relPath(datasetName, 'spectrogram', [s.id '.png']);
meta.metadata_path = relPath(datasetName, 'metadata', [s.id '.json']);
meta.sample_rate_hz = s.fs;
meta.duration_s = s.dur;
meta.source = source;
meta.ssp = struct('depth_m', s.z(:).', 'temperature_c', s.temp(:).', ...
    'sound_speed_mps', c(:).');
if isfield(s, 'woa')
    meta.woa = s.woa;
end
meta.channel = struct('simulator', 'Bellhop', ...
    'env_path', relPath(datasetName, 'channel', [s.id '.env']), ...
    'arr_path', relPath(datasetName, 'channel', [s.id '.arr']), ...
    'source_depth_m', s.sd, 'receiver_depth_m', s.rd, 'range_km', s.rr, ...
    'bellhop_frequency_hz', fc, 'multipath_count', numel(delayS), ...
    'path_delay_s', delayS(:).', 'path_amplitude_real', real(amp(:)).', ...
    'path_amplitude_imag', imag(amp(:)).');
meta.receiver = noiseInfo;
if isfield(s, 'class_label'), meta.class_label = s.class_label; end
if isfield(s, 'group_id'), meta.group_id = s.group_id; end
if isfield(s, 'split'), meta.split = s.split; end

writeText(json, jsonencode(meta));
end

function path = relPath(datasetName, folder, file)
path = fullfile('output', 'datasets', datasetName, folder, file);
end

function noiseDir = getNoiseDir(s, dataRoot)
% 可在参数结构体中指定 s.noise_dir；默认使用 SRTP 根目录下的环境噪声文件夹。
if isfield(s, 'noise_dir')
    noiseDir = s.noise_dir;
    return;
end
workspaceDir = fileparts(fileparts(fileparts(dataRoot)));
noiseDir = fullfile(fileparts(workspaceDir), '环境噪声', '环境噪声');
end

function saveSpectrogram(x, fs, path, px)
% Fixed square image without text, axes, or colour bar to avoid label leakage.
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 px px]);
spectrogram(x, hann(512, 'periodic'), 448, 1024, fs, 'yaxis');
ax = gca;
ylim(ax, [0 fs/2] / 1000);
axis(ax, 'off');
set(ax, 'Units', 'normalized', 'Position', [0 0 1 1]);
exportgraphics(ax, path, 'Resolution', 96);
close(fig);
end

function value = getField(s, name, default)
if isfield(s, name), value = s.(name); else, value = default; end
end



function writeText(path, text)
fid = fopen(path, 'w');
assert(fid ~= -1, 'Cannot write %s.', path);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end
