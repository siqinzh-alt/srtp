function [rx, delayS, amp, alpha] = delayandsum_doppler(tx, fs, Arr, v, c0, ir, ird, isd)
%DELAYANDSUM_DOPPLER 原版 BELLHOP delayandsum 的可调用多普勒版本。
%   v = [vr vz]，单位 m/s；vr 为径向速度，vz 为垂向速度。
%   仅复制并改写原版脚本到函数形式，原版目录文件未修改。

if nargin < 4 || isempty(v), v = [0 0]; end
if nargin < 5 || isempty(c0), c0 = 1500; end
if nargin < 6, ir = 1; end
if nargin < 7, ird = 1; end
if nargin < 8, isd = 1; end
assert(numel(v) == 2, 'receiver_velocity_mps 必须是 [vr vz]。');

tx = tx(:);
n = numel(tx);
nPath = Arr.Narr(ir, ird, isd);
assert(nPath > 0, 'No Bellhop arrivals exist at the selected receiver.');

amp = squeeze(Arr.A(ir, 1:nPath, ird, isd));
delayS = real(squeeze(Arr.delay(ir, 1:nPath, ird, isd)));
angle = squeeze(Arr.RcvrAngle(ir, 1:nPath, ird, isd));
theta = angle(:) * pi / 180;
rayTangent = [cos(theta), sin(theta)];
alpha = 1 - rayTangent * (v(:) / c0);
assert(all(alpha > 0), '多普勒因子必须为正，请检查接收机速度。');

if norm(v) == 0
    txHilbert = hilbert(tx);
    txDoppler = txHilbert;
    index = ones(nPath, 1);
else
    % 保留原版预计算多个 Doppler 波形、再选最近波形的做法。
    nAlpha = 51;
    alphaGrid = linspace(1 - norm(v)/c0, 1 + norm(v)/c0, nAlpha);
    txDoppler = zeros(n, nAlpha);
    for k = 1:nAlpha
        txDoppler(:, k) = hilbert(arbitrary_alpha(tx.', alphaGrid(k)).');
    end
    index = 1 + round((alpha - alphaGrid(1)) / ...
        (alphaGrid(end) - alphaGrid(1)) * (nAlpha - 1));
    index = min(max(index, 1), nAlpha);
end

rx = zeros(n, 1);
for k = 1:nPath
    shift = round(delayS(k) * fs);
    if shift < n
        pathSignal = zeros(n, 1);
        pathSignal(shift+1:end) = real(amp(k) .* ...
            txDoppler(1:end-shift, index(k)));
        rx = rx + pathSignal;
    end
end
end
