function [x, label, meta] = generate_noncooperative_bpsk(p, fs, dur)
%GENERATE_NONCOOPERATIVE_BPSK Create a random-payload BPSK burst.
%   This is a noncooperative-communication model: bits are random rather
%   than a known repeated m-sequence. It calls the local Bellhop bpsk.m.
%
% Required p fields: carrier_hz, chips_per_sec (or chip_rate_hz),
%                    start_time_s, pulse_width_s.
% Optional p fields: bits (0/1 vector), amplitude (default 0.95),
%                    fade_s (default 0).

required = {'carrier_hz','start_time_s','pulse_width_s'};
for k = 1:numel(required)
    assert(isfield(p, required{k}), 'Missing BPSK parameter: %s.', required{k});
end
if isfield(p, 'chips_per_sec')
    chipRate = p.chips_per_sec;
elseif isfield(p, 'chip_rate_hz')
    chipRate = p.chip_rate_hz;
else
    error('BPSK needs chips_per_sec (the Bellhop bpsk.m parameter).');
end

fc = p.carrier_hz;
start = p.start_time_s;
width = p.pulse_width_s;
amp = fieldOr(p, 'amplitude', 0.95);
fade = fieldOr(p, 'fade_s', 0);
assert(fc > 0 && fc < fs/2, 'carrier_hz must lie between 0 and fs/2.');
assert(chipRate > 0 && fs/chipRate == fix(fs/chipRate), ...
    'fs/chips_per_sec must be an integer, as required by bpsk.m.');
assert(start >= 0 && width > 0 && start + width <= dur, ...
    'The BPSK burst must fit inside the clip.');

samplesPerChip = fs/chipRate;
activeSamples = round(width*fs);
assert(mod(activeSamples, samplesPerChip) == 0, ...
    'pulse_width_s*fs must contain an integer number of BPSK chips.');
numBits = activeSamples/samplesPerChip;

if isfield(p, 'bits')
    bits = p.bits(:);
    assert(numel(bits) == numBits && all(bits == 0 | bits == 1), ...
        'bits must be a 0/1 vector containing exactly %d bits.', numBits);
    payloadSource = 'provided_bits';
else
    bits = randi([0 1], numBits, 1);
    payloadSource = 'random_bits';
end

bipolar = 2*bits - 1;
burst = amp * bpsk(bipolar.', fc, fs, chipRate).';
burst = applyFade(burst, fs, fade);

n = round(dur*fs);
x = zeros(n, 1);
first = round(start*fs) + 1;
x(first:first+activeSamples-1) = burst;

label = 'BPSK_NONCOOPERATIVE';
meta = struct('type', 'BPSK', 'modulation', 'BPSK', ...
    'application', 'noncooperative_communication', ...
    'carrier_hz', fc, 'chips_per_sec', chipRate, ...
    'samples_per_chip', samplesPerChip, 'bit_count', numBits, ...
    'payload_source', payloadSource, 'amplitude', amp, 'fade_s', fade, ...
    'start_time_s', start, 'pulse_width_s', width);
end

function value = fieldOr(s, name, default)
if isfield(s, name), value = s.(name); else, value = default; end
end

function y = applyFade(y, fs, fade)
edge = min(round(fade*fs), floor(numel(y)/2));
if edge == 0, return; end
ramp = 0.5*(1-cos(pi*(0:edge-1)'/edge));
y(1:edge) = y(1:edge).*ramp;
y(end-edge+1:end) = y(end-edge+1:end).*flipud(ramp);
end
