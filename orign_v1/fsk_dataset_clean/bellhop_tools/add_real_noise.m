function [rx, info] = add_real_noise(clean, fs, targetSnrDb, noiseDir)
%ADD_REAL_NOISE 从真实环境录音随机截取片段，按目标 SNR 叠加到 clean。

assert(exist(noiseDir, 'dir') == 7, '环境噪声文件夹不存在：%s', noiseDir);
files = dir(fullfile(noiseDir, '*.wav'));
assert(~isempty(files), '环境噪声文件夹中没有 WAV 文件：%s', noiseDir);

file = files(randi(numel(files)));
noisePath = fullfile(file.folder, file.name);
audio = audioinfo(noisePath);
nTarget = numel(clean);
nRead = max(2, ceil(nTarget * audio.SampleRate / fs) + 4);

if audio.TotalSamples >= nRead
    startSample = randi(audio.TotalSamples - nRead + 1);
    raw = audioread(noisePath, [startSample, startSample + nRead - 1]);
else
    startSample = 1;
    raw = audioread(noisePath);
end

if size(raw, 2) > 1
    raw = mean(raw, 2);                 % 多通道录音转为单通道
end
raw = raw - mean(raw);                  % 去掉录音的直流偏置
if audio.SampleRate ~= fs
    raw = resample(raw, fs, audio.SampleRate);
end
if numel(raw) < nTarget
    raw = repmat(raw, ceil(nTarget / numel(raw)), 1);
end
noise = raw(1:nTarget);

cleanRms = sqrt(mean(clean.^2));
noiseRms = sqrt(mean(noise.^2));
assert(cleanRms > eps, '传播后的信号能量为零，无法按 SNR 加噪。');
assert(noiseRms > eps, '选取的环境噪声片段能量为零。');
noise = noise / noiseRms * cleanRms / sqrt(10^(targetSnrDb / 10));
rx = clean + noise;

info = struct( ...
    'noise_type', 'real_environment_recording', ...
    'target_snr_db', targetSnrDb, ...
    'actual_snr_db', 10 * log10(mean(clean.^2) / mean(noise.^2)), ...
    'noise_file', fullfile('环境噪声', '环境噪声', file.name), ...
    'noise_start_sample', startSample, ...
    'noise_source_sample_rate_hz', audio.SampleRate, ...
    'noise_resampled_to_hz', fs, ...
    'noise_source_channels', audio.NumChannels);
end
