function [rx, delayS, amp] = delayandsum(tx, fs, Arr, ir, ird, isd)
%DELAYANDSUM Create a received waveform from Bellhop multipath arrivals.
% Adapted from Bellhop's legacy delayandsum.m into a callable function.

if nargin < 4, ir = 1; end
if nargin < 5, ird = 1; end
if nargin < 6, isd = 1; end
tx = tx(:);
nPath = Arr.Narr(ir, ird, isd);
assert(nPath > 0, 'No Bellhop arrivals exist at the selected receiver.');

amp = squeeze(Arr.A(ir, 1:nPath, ird, isd));
delayS = real(squeeze(Arr.delay(ir, 1:nPath, ird, isd)));
txAnalytic = hilbert(tx);
rx = zeros(size(tx));

for k = 1:nPath
    shift = round(delayS(k)*fs);
    assert(shift >= 0, 'Negative arrival delays are not supported.');
    if shift < numel(tx)
        pathSignal = zeros(size(tx));
        pathSignal(shift+1:end) = real(amp(k).*txAnalytic(1:end-shift));
        rx = rx + pathSignal;
    end
end
end
