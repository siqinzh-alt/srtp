function [x, label, meta] = generate_signal(p, fs, dur)
%GENERATE_SIGNAL Create a source waveform independently of Bellhop.
%   p.type supports 'CW', 'LFM', '2FSK', and '4FSK'.  All signals are
%   real-valued column vectors, length round(fs*dur), with a quiet region
%   outside p.start_time_s to p.start_time_s + p.pulse_width_s.

arguments
    p struct
    fs (1,1) double {mustBePositive}
    dur (1,1) double {mustBePositive}
end

kind = upper(string(need(p, 'type')));
start = need(p, 'start_time_s');
width = need(p, 'pulse_width_s');
assert(start >= 0 && width > 0 && start + width <= dur, ...
    'The active signal interval must lie within the clip.');

n = round(fs * dur);
t = (0:n-1)' / fs;
active = t >= start & t < start + width;
ta = t(active) - start;
x = zeros(n, 1);
fade = fieldOr(p, 'fade_s', 0.02);

switch kind
    case "CW"
        fc = need(p, 'carrier_hz');
        assert(fc > 0 && fc < fs/2, 'carrier_hz must be between 0 and fs/2.');
        y = sin(2*pi*fc*ta);
        label = 'CW';
        meta = struct('type', 'CW', 'carrier_hz', fc, ...
            'start_time_s', start, 'pulse_width_s', width, 'fade_s', fade);

    case "LFM"
        f0 = need(p, 'f_start_hz');
        f1 = need(p, 'f_end_hz');
        assert(all([f0 f1] > 0) && all([f0 f1] < fs/2), ...
            'LFM frequencies must be between 0 and fs/2.');
        k = (f1-f0) / width;
        y = sin(2*pi*(f0*ta + 0.5*k*ta.^2));
        label = 'LFM';
        meta = struct('type', 'LFM', 'f_start_hz', f0, 'f_end_hz', f1, ...
            'start_time_s', start, 'pulse_width_s', width, 'fade_s', fade);

    case {"2FSK", "4FSK"}
        tones = need(p, 'tone_hz');
        m = sscanf(char(kind), '%dFSK');
        assert(numel(tones) == m && all(tones > 0) && all(tones < fs/2), ...
            'tone_hz must contain %d valid tones for %s.', m, kind);
        rs = need(p, 'symbol_rate_hz');
        ns = max(1, round(width * rs));
        sym = fieldOr(p, 'symbols', randi([0 m-1], ns, 1));
        sym = sym(:);
        assert(numel(sym) == ns && all(sym >= 0) && all(sym < m), ...
            'symbols must contain %d integers from 0 to %d.', ns, m-1);
        symbolIndex = min(floor(ta*rs)+1, ns);
        f = tones(sym(symbolIndex)+1);
        y = sin(2*pi*cumsum(f)/fs); % phase-continuous FSK
        label = char(kind);
        meta = struct('type', label, 'tone_hz', tones(:).', ...
            'symbol_rate_hz', rs, 'symbols', sym.', 'start_time_s', start, ...
            'pulse_width_s', width, 'fade_s', fade);

    otherwise
        error('Unsupported signal type: %s. Use CW, LFM, 2FSK, or 4FSK.', kind);
end

x(active) = applyFade(y, fs, fade);
end

function value = need(s, name)
assert(isfield(s, name), 'Signal parameter "%s" is required.', name);
value = s.(name);
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
