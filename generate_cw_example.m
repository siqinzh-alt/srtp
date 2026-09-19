%% GENERATE_CW_EXAMPLE

clear; clc; close all;

%% 路径与本地 Bellhop 依赖配置
root = fileparts(mfilename('fullpath'));
tool = fullfile(root, 'bellhop_tools');
out = fullfile(root, 'output', 'cw_bellhop_demo');

if ~exist(out, 'dir'), mkdir(out); end
% 把本地工具包置顶加入搜索路径，防止被其他版本的 Bellhop 同名函数遮蔽
addpath(tool, '-begin');

%% Dataset parameters
id = 'cw_bellhop_000001';
rng(20260919, 'twister');
fs = 16000;                 % Hz
dur = 5;                    % seconds
snrDb = 10;                 % received SNR (dB)
makePlots = true;           % 是否保存诊断图（SSP/到达/声线图/频谱图）

%%make signal
sig = struct('type', 'CW', 'carrier_hz', 3200, ...
    'start_time_s', 1.0, 'pulse_width_s', 2.5, 'fade_s', 0.02);
fc = signal_center_frequency(sig); % Bellhop centre frequency (Hz)

%env
z = [0; 10; 20; 40; 60; 80; 100];
temp = [25; 24.2; 22.8; 19.5; 16.8; 15.4; 15.0];
salt = [34.2; 34.3; 34.5; 34.6; 34.7; 34.8; 34.8];
sd = 20;                     % source depth (m)
rd = 50;                     % receiver depth (m)
rr = 1.0;                    % receiver range (km)

assert(fc < fs/2, 'fc must be below the Nyquist frequency.'); % 载频必须低于奈奎斯特频率，防混叠

%% 1. Build the structures 

[SSP, Bdry, Pos, Beam, cInt, RMax, c] = build_ssp(z, temp, salt, sd, rd, rr);

%% 2. 写入器 + 求解器 + 到达读取器
env = fullfile(out, [id '.env']);
arr = fullfile(out, [id '.arr']);
write_env(env, 'BELLHOP', 'CW sample with T/S/depth SSP', fc, ...
    SSP, Bdry, Pos, Beam, cInt, RMax);

old = pwd;
restore = onCleanup(@() cd(old));
cd(out);                     % Bellhop writes files into its current folder.
bellhop(id);                 % local bellhop.m + bellhop.exe
assert(isfile(arr), 'Bellhop did not generate %s.', arr);

%解析 .arr：Arr 含各路径的时延和复振幅
[Arr, ~] = read_arrivals_local(arr, 500); % local copy, avoids name collisions

%% 3. Generate the transmitted signal.
[tx, label, source] = generate_signal(sig, fs, dur);

%% 4. Multipath synthesis and receiver noise.

[clean, delayS, amp] = delayandsum(tx, fs, Arr, 1, 1, 1);
noise = makenoise(fc, 7000, dur, fs);
noise = noise/sqrt(mean(noise.^2));
noise = noise*sqrt(mean(clean.^2)/10^(snrDb/10));
rx = clean + noise;
actualSnrDb = 10*log10(mean(clean.^2)/mean((rx-clean).^2));
if max(abs(rx)) > 0.999, rx = 0.999*rx/max(abs(rx)); end

%% 5. Final dataset files and relative-path metadata.
wav = fullfile(out, [id '.wav']);
json = fullfile(out, [id '.json']);
manifest = fullfile(out, 'manifest.jsonl');
png = fullfile(out, [id '.png']);
audiowrite(wav, rx, fs, 'BitsPerSample', 16);

meta.id = id;
meta.label = label;
meta.audio_path = fullfile('output', 'cw_bellhop_demo', [id '.wav']);
meta.sample_rate_hz = fs;
meta.duration_s = dur;
meta.source = source;
meta.ssp = struct('depth_m', z.', 'temperature_c', temp.', ...
    'salinity_psu', salt.', 'sound_speed_mps', c.');
meta.channel = struct('simulator', 'Bellhop', ...
    'env_path', fullfile('output', 'cw_bellhop_demo', [id '.env']), ...
    'arr_path', fullfile('output', 'cw_bellhop_demo', [id '.arr']), ...
    'source_depth_m', sd, 'receiver_depth_m', rd, 'range_km', rr, ...
    'multipath_count', numel(delayS), 'path_delay_s', delayS.', ...
    'path_amplitude_real', real(amp).', 'path_amplitude_imag', imag(amp).');
meta.receiver = struct('noise_type', 'bandpass_gaussian', ...
    'target_snr_db', snrDb, 'actual_snr_db', actualSnrDb);

line = jsonencode(meta);
writeLine(json, line, 'w');
writeLine(manifest, line, 'w');

figure('Color', 'w', 'Position', [100 100 900 450]);
spectrogram(rx, hann(512, 'periodic'), 448, 1024, fs, 'yaxis');
ylim([0 fs/2]/1000); colorbar;
title(sprintf('Bellhop CW: %.0f Hz, %.1f dB SNR', fc, actualSnrDb));
xlabel('Time (s)'); ylabel('Frequency (kHz)');
exportgraphics(gcf, png, 'Resolution', 160);

if makePlots
    % These are the diagnostic plots relevant to this flat-bottom,
    % range-independent SSP example.  They supplement the spectrogram above.
    plot_ssp(SSP, fullfile(out, [id '_ssp.png']));
    plotarr_local(delayS, amp, Pos, fullfile(out, [id '_arrivals.png']));

    % An A run produces arrivals; an additional R run produces ray paths.
    rayId = [id '_ray'];
    rayBeam = Beam;
    rayBeam.RunType = 'R';
    write_env(fullfile(out, [rayId '.env']), 'BELLHOP', 'CW ray path', fc, ...
        SSP, Bdry, Pos, rayBeam, cInt, RMax);
    bellhop(rayId);
    ray = fullfile(out, [rayId '.ray']);
    if isfile(ray)
        global units jkpsflag
        units = 'km';
        jkpsflag = false;
        figure('Color', 'w', 'Position', [100 100 900 450]);
        plotray([rayId '.ray']);
        exportgraphics(gcf, fullfile(out, [id '_rays.png']), 'Resolution', 160);
    else
        warning('Bellhop did not generate the ray diagnostic file: %s', ray);
    end
end

fprintf('Created final sample:\n%s\n%s\n%s\n', wav, json, manifest);

function writeLine(path, text, mode)
fid = fopen(path, mode);
assert(fid ~= -1, 'Cannot write %s.', path);
closeFile = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', text);
end
