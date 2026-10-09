function [rx, noiseInfo] = add_noise(clean, s)

% add_real_noise 会按噪声原始采样率读取足够长的片段，
% 重采样到 s.fs，再截取或循环到与 clean 相同的 3 s 长度，并按 s.snr_db 缩放。
[rx, noiseInfo] = add_real_noise(clean, s.fs, s.snr_db, s.noise_dir);
noiseInfo.source_split = s.noise_split;
assert(numel(rx) == round(s.fs * s.dur), ...
    '加噪后的信号长度与设定总时长不一致。');
if max(abs(rx)) > 0.999, rx = 0.999 * rx / max(abs(rx)); end
end
