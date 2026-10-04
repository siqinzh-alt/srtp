function samples = make_lfm_parameter_set(n)
%MAKE_LFM_PARAMETER_SET Sample diverse, physically valid LFM conditions.
%   The random generator is set by the caller.  Every sampled value is
%   retained in the per‑sample JSON and manifest, so the dataset remains
%   reproducible and auditable.

filename = "E:\Matlab\bin\cw_dataset_clean\bellhop_tools\woa23_decav_t01mn01.csv";
lines = readlines(filename);
depthLine = lines(2);                        % 深度注释行
depthStr = strtrim(extractAfter(depthLine, ':'));
depth_vec = str2double(strsplit(depthStr, ','));
depth_vec = depth_vec(:);

M = readmatrix(filename);                    % M第1行=第一个数据点
lat_all = M(:, 1);                           % 第1列 = 纬度
lon_all = M(:, 2);                           % 第2列 = 经度
temp_all = M(:, 3:end);                      % 第3列往后 = 各深度温度

%目标坐标：109°32′35″E，17°52′02″N
tarlon = 109 + 32/60 + 35/3600;
tarlat = 17 + 52/60 + 2/3600;

dis1 = (tarlon - lon_all).^2;
dis2 = (tarlat - lat_all).^2;
dis = dis1 + dis2;
[~, index] = min(dis);

temp_vec = temp_all(index, :).';
keep = ~isnan(depth_vec) & ~isnan(temp_vec);
z = depth_vec(keep);
temp = temp_vec(keep);
zMax = z(end);
disp(['读取成功：深度层数=', num2str(length(z)), '，最大水深=', num2str(zMax), 'm']);

samples = repmat(struct(), n, 1);

for k = 1:n
    dur = 3;
    pulse = 1 + 1.2*rand;              % 脉冲宽度 1.4‑2.8 s
    start = 0.4 + (dur-pulse-0.8)*rand;  % 脉冲起始时间，前后留静音

    f_start = 2000 + 6000*rand;
    f_end   = 2000 + 6000*rand;

    if f_end < f_start
        chirp_dir = 'down';
        bw_hz = f_start - f_end;
    else
        chirp_dir = 'up';
        bw_hz = f_end - f_start;
    end

    %盐度剖面，与深度长度匹配
    salt = linspace(33.8 + 0.4*rand, 34.6 + 0.35*rand, length(z)).';

    %收发深度，避开海面、海底，取浮点数
    sd = 0.1 + (zMax - 0.2)*rand;
    rd = 0.1 + (zMax - 0.2)*rand;

    samples(k).id = sprintf('lfm_%06d', k);
    samples(k).sig = struct(...
        'type', 'LFM', ...
        'f_start', f_start, ...
        'f_end', f_end, ...
        'bw_hz', bw_hz, ...
        'chirp_dir', chirp_dir, ...
        'start_time_s', start, ...
        'pulse_width_s', pulse, ...
        'fade_s', 0.02);

    samples(k).fs = 16000;
    samples(k).dur = dur;
    samples(k).snr_db = randi([0 15]);
    samples(k).z = z;
    samples(k).temp = temp;
    samples(k).salt = salt;
    samples(k).sd = sd;
    samples(k).rd = rd;
    samples(k).rr = 0.5 + 1.0*rand;      % 收发水平距离 km
end
end
