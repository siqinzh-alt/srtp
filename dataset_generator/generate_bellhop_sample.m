function meta = generate_bellhop_sample(s, dataRoot, datasetName)
%GENERATE_BELLHOP_SAMPLE Produce one model-ready waveform and its records.
%   Outputs are placed in audio/, spectrogram/, metadata/, and channel/.
%   All paths written to metadata are relative to the workspace root.
audioDir = fullfile(dataRoot, 'audio');
specDir = fullfile(dataRoot, 'spectrogram');
metaDir = fullfile(dataRoot, 'metadata');
channelDir = fullfile(dataRoot, 'channel');
dirs = {audioDir, specDir, metaDir, channelDir};
dirs = {audioDir, specDir, metaDir, channelDir};
for k = 1:numel(dirs)
if ~exist(dirs{k}, 'dir'), mkdir(dirs{k}); end
end
fc = signal_center_frequency(s.sig);
%构建声速剖面
[SSP, Bdry, Pos, Beam, cInt, RMax, c] = ...
    build_ssp(s.z, s.temp, s.salt, s.sd, s.rd, s.rr);
env = fullfile(channelDir, [s.id '.env']);
arr = fullfile(channelDir, [s.id '.arr']);
write_env(env, 'BELLHOP', s.id, fc, SSP, Bdry, Pos, Beam, cInt, RMax);
old = pwd;%`pwd` = print working directory，获取当前 MATLAB 工作文件夹的完整路径
restoreFolder = onCleanup(@() cd(old));%onCleanup：只要对象被清理就执行括号内函数
cd(channelDir);
bellhop(s.id);
assert(isfile(arr), 'Bellhop did not generate %s.', arr);
[Arr, ~] = read_arrivals_local(arr, 500);
[tx, label, source] = generate_signal(s.sig, s.fs, s.dur);%生成CW 脉冲时域信号
[clean, delayS, amp] = delayandsum(tx, s.fs, Arr, 1, 1, 1);
noiseFile = "C:\Users\asus\Desktop\81__25_09_13_H3_corriente.wav";  % 噪声文件名
[noise_raw, fs_noise] = audioread(noiseFile);
noise_raw = noise_raw(:,1);
noise_raw = resample(noise_raw, s.fs, fs_noise);
N = length(clean);
if length(noise_raw) < N
    noise_raw = repmat(noise_raw, ceil(N/length(noise_raw)), 1);
end
noise_raw = noise_raw(1:N);
noise = noise_raw / sqrt(mean(noise_raw.^2));
noise = noise * sqrt(mean(clean.^2) / 10^(s.snr_db/10));
% =================================================================
rx = clean + noise;
actualSnrDb = 10*log10(mean(clean.^2) / mean((rx-clean).^2));
if max(abs(rx)) > 0.999, rx = 0.999*rx/max(abs(rx)); end
wav = fullfile(audioDir, [s.id '.wav']);
png = fullfile(specDir, [s.id '.png']);
json = fullfile(metaDir, [s.id '.json']);
audiowrite(wav, rx, s.fs, 'BitsPerSample', 16);
figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 450]);
spectrogram(rx, hann(512, 'periodic'), 448, 1024, s.fs, 'yaxis');
ylim([0 s.fs/2]/1000); colorbar;
title(sprintf('%s: %.0f Hz, %.1f dB SNR', label, fc, actualSnrDb));
xlabel('Time (s)'); ylabel('Frequency (kHz)');
exportgraphics(gcf, png, 'Resolution', 160);
close(gcf);
meta.id = s.id;
meta.label = label;
meta.audio_path = relPath(datasetName, 'audio', [s.id '.wav']);
meta.spectrogram_path = relPath(datasetName, 'spectrogram', [s.id '.png']);
meta.metadata_path = relPath(datasetName, 'metadata', [s.id '.json']);
meta.sample_rate_hz = s.fs;
meta.duration_s = s.dur;
meta.source = source;
meta.ssp = struct('depth_m', s.z(:).', 'temperature_c', s.temp(:).', ...
'salinity_psu', s.salt(:).', 'sound_speed_mps', c(:).');
meta.channel = struct('simulator', 'Bellhop', ...
'env_path', relPath(datasetName, 'channel', [s.id '.env']), ...
'arr_path', relPath(datasetName, 'channel', [s.id '.arr']), ...
'source_depth_m', s.sd, 'receiver_depth_m', s.rd, 'range_km', s.rr, ...
'bellhop_frequency_hz', fc, 'multipath_count', numel(delayS), ...
'path_delay_s', delayS(:).', 'path_amplitude_real', real(amp(:)).', ...
'path_amplitude_imag', imag(amp(:)).');
meta.receiver = struct('noise_type', 'real_underwater_noise', ...
'target_snr_db', s.snr_db, ...
'actual_snr_db', actualSnrDb);

meta.qa = struct();
meta.qa.question = "这是什么信号？";
meta.qa.choices = {"cw","lfm","fsk","bpsk","船舶辐射噪声"};
meta.qa.answer = "cw";

writeText(json, jsonencode(meta));
end

function path = relPath(datasetName, folder, file)
path = fullfile('output', 'datasets', datasetName, folder, file);
end

function writeText(path, text)
fid = fopen(path, 'w');
assert(fid ~= -1, 'Cannot write %s.', path);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end
