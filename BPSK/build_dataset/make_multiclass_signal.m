function s = make_multiclass_signal(className)
%MAKE_MULTICLASS_SIGNAL Draw parameters and a source waveform for one class.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'bellhop_tools'), '-begin');

fs = 16000;
dur = 3;
carrierCandidates = [2200, 2600, 3000, 3400, 3800, 4200, 4600];

s.fs = fs;
s.dur = dur;
s.snr_db = randi([0, 15]);
s.image_px = 448;

switch upper(className)
    case 'BPSK'
        chipRateCandidates = [250, 500, 800, 1000, 1600, 2000];
        chipRate = chipRateCandidates(randi(numel(chipRateCandidates)));
        pulse = 1.4 + 1.4 * rand;
        pulse = round(pulse * chipRate) / chipRate;
        start = randomStart(dur, pulse);
        s.sig = struct('type', 'BPSK', 'chips_per_sec', chipRate, ...
            'carrier_hz', carrierCandidates(randi(numel(carrierCandidates))), ...
            'start_time_s', start, 'pulse_width_s', pulse, 'fade_s', 0.02);

    case 'CW'
        pulse = 1.4 + 1.4 * rand;
        s.sig = struct('type', 'CW', ...
            'carrier_hz', carrierCandidates(randi(numel(carrierCandidates))), ...
            'start_time_s', randomStart(dur, pulse), ...
            'pulse_width_s', pulse, 'fade_s', 0.02);

    case 'LFM'
        center = carrierCandidates(randi(numel(carrierCandidates)));
        bandwidthCandidates = [500, 800, 1000, 1200];
        bandwidth = bandwidthCandidates(randi(numel(bandwidthCandidates)));
        f0 = max(300, center - bandwidth / 2);
        f1 = min(fs / 2 - 300, center + bandwidth / 2);
        if rand < 0.5, [f0, f1] = deal(f1, f0); end
        pulse = 1.4 + 1.4 * rand;
        s.sig = struct('type', 'LFM', 'f_start_hz', f0, 'f_end_hz', f1, ...
            'start_time_s', randomStart(dur, pulse), ...
            'pulse_width_s', pulse, 'fade_s', 0.02);

    case 'FSK'
        center = carrierCandidates(randi(numel(carrierCandidates)));
        spacingCandidates = [250, 400, 500, 800];
        spacing = spacingCandidates(randi(numel(spacingCandidates)));
        tones = center + [-spacing / 2, spacing / 2];
        pulse = 1.4 + 1.4 * rand;
        symbolRateCandidates = [50, 100, 200, 250];
        s.sig = struct('type', '2FSK', 'tone_hz', tones, ...
            'symbol_rate_hz', symbolRateCandidates(randi(numel(symbolRateCandidates))), ...
            'start_time_s', randomStart(dur, pulse), ...
            'pulse_width_s', pulse, 'fade_s', 0.02);

    otherwise
        error('Unsupported class: %s', className);
end

[s.tx, s.label, s.source] = generate_signal(s.sig, s.fs, s.dur);
end

function start = randomStart(dur, pulse)
start = 0.1 + (dur - pulse - 0.2) * rand;
end
