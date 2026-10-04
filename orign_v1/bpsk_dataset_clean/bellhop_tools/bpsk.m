function s = bpsk( s_bipolar, fc, fs, chips_per_sec )
%BPSK Bellhop waveform helper: encode a bipolar sequence with BPSK.
%   Parameters follow the Bellhop waveform-library version.

samples_per_chip = fs / chips_per_sec;
if samples_per_chip ~= fix(samples_per_chip)
    error('samples_per_chip must be an integer: fs/chips_per_sec.');
end

deltat = 1 / fs;
tsin = 0:deltat:(samples_per_chip-1)*deltat;
sinwave = sin(2*pi*fc*tsin)';
s = sinwave * s_bipolar(:).';
s = reshape(s, 1, []);
end
