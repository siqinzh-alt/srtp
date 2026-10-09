function [x, source] = make_ship_noise(noiseDir, fs, dur)
%MAKE_SHIP_NOISE Extract a normalized clip from a real ShipsEar recording.

files = dir(fullfile(noiseDir, '*.wav'));
files = files(~[files.isdir]);
assert(~isempty(files), '船舶噪声文件夹中没有 WAV 文件：%s', noiseDir);

file = files(randi(numel(files)));
path = fullfile(file.folder, file.name);
info = audioinfo(path);
nTarget = round(fs * dur);
nRead = max(2, ceil(nTarget * info.SampleRate / fs) + 4);

if info.TotalSamples >= nRead
    startSample = randi(info.TotalSamples - nRead + 1);
    raw = audioread(path, [startSample, startSample + nRead - 1]);
else
    startSample = 1;
    raw = audioread(path);
end
if size(raw, 2) > 1, raw = mean(raw, 2); end
raw = raw - mean(raw);
if info.SampleRate ~= fs, raw = resample(raw, fs, info.SampleRate); end
if numel(raw) < nTarget, raw = repmat(raw, ceil(nTarget / numel(raw)), 1); end
x = raw(1:nTarget);
x = 0.85 * x / max(sqrt(mean(x.^2)), eps);
x = max(min(x, 0.999), -0.999);

[~, splitName] = fileparts(noiseDir);
source = struct('type', 'SHIP_NOISE', 'recording_file', ...
    fullfile(splitName, file.name), 'recording_start_sample', startSample, ...
    'recording_sample_rate_hz', info.SampleRate, 'output_sample_rate_hz', fs, ...
    'recording_channels', info.NumChannels);
end
