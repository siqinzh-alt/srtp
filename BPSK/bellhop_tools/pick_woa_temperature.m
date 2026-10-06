function [z, temp, info] = pick_woa_temperature(woaDir, zMax)
%PICK_WOA_TEMPERATURE Randomly select one WOA temperature-depth profile.

assert(exist(woaDir, 'dir') == 7, 'WOA 文件夹不存在：%s', woaDir);
files = dir(fullfile(woaDir, '**', '*.csv'));
files = files(~[files.isdir]);

isTemp = false(size(files));
for k = 1:numel(files)
    id = fopen(fullfile(files(k).folder, files(k).name), 'r');
    assert(id ~= -1, '无法读取 WOA 文件：%s', files(k).name);
    first = fgetl(id);
    fclose(id);
    isTemp(k) = ischar(first) && contains(lower(first), 'temperature');
end
files = files(isTemp);
assert(~isempty(files), ...
    ['WOA 中没有温度文件。当前 s13--s16 是盐度文件；' ...
     '请放入表头含 temperature 的 WOA t13--t16 CSV 文件。']);

file = files(randi(numel(files)));
path = fullfile(file.folder, file.name);
id = fopen(path, 'r');
assert(id ~= -1, '无法读取 WOA 文件：%s', path);
fgetl(id);
second = fgetl(id);
fclose(id);

token = regexp(second, 'DEPTHS \(M\):(.*)$', 'tokens', 'once');
assert(~isempty(token), '无法从 WOA 文件读取深度：%s', path);
depthCell = textscan(token{1}, '%f', 'Delimiter', ',');
zAll = depthCell{1}.';
raw = readmatrix(path, 'NumHeaderLines', 2);
assert(size(raw, 2) >= numel(zAll) + 2, 'WOA 数据列不完整：%s', path);

zMask = zAll <= zMax;
z = zAll(zMask).';
values = raw(:, 3:2 + numel(zAll));
values = values(:, zMask);
validRows = sum(isfinite(values), 2) >= 2;
assert(any(validRows), 'WOA 文件在 %.0f m 内没有完整温度剖面：%s', zMax, path);
rows = find(validRows);
row = rows(randi(numel(rows)));

temp = values(row, :).';
good = isfinite(temp);
temp = interp1(z(good), temp(good), z, 'linear', 'extrap');
info = struct('source', 'WOA23', 'file', file.name, ...
    'latitude_deg', raw(row, 1), 'longitude_deg', raw(row, 2), ...
    'max_depth_m', zMax);
end
