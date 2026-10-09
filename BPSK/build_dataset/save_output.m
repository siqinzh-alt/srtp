function meta = save_output(rx, s, dataRoot, datasetName)
%SAVE_OUTPUT 输出 WAV 和时频图，并返回 QA 标注所需字段。

dirs.audio = fullfile(dataRoot, 'audio');
dirs.spectrogram = fullfile(dataRoot, 'spectrogram');

for path = {dirs.audio, dirs.spectrogram}
    if ~exist(path{1}, 'dir'), mkdir(path{1}); end
end

wav = fullfile(dirs.audio, [s.id '.wav']);
png = fullfile(dirs.spectrogram, [s.id '.png']);
audiowrite(wav, rx, s.fs, 'BitsPerSample', 16);
saveSpectrogram(rx, s.fs, png, s.image_px);

% 仅在内存中构造 QA 信息，不生成单独的元数据 JSON 文件。
meta = struct('id', s.id, 'class_label', s.class_label, ...
    'spectrogram_path', relPath(datasetName, 'spectrogram', [s.id '.png']));
end

function saveSpectrogram(x, fs, path, px)
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

function path = relPath(datasetName, folder, file)
path = fullfile('output', 'datasets', datasetName, folder, file);
end

