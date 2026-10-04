function fc = signal_center_frequency(p)
%SIGNAL_CENTER_FREQUENCY Select the single frequency used in this Bellhop run.
%   A time-domain waveform can be broadband, but this basic data generator
%   uses one Bellhop arrival set.  Therefore LFM/FSK use their band centre.
%   Set p.bellhop_frequency_hz to explicitly override that choice.

if isfield(p, 'bellhop_frequency_hz')
    fc = p.bellhop_frequency_hz;
    return
end

switch upper(string(p.type))
    case "CW"
        fc = p.carrier_hz;
    case "LFM"
        fc = mean([p.f_start_hz, p.f_end_hz]);
    case {"2FSK", "4FSK"}
        fc = mean(p.tone_hz);
    otherwise
        error('Cannot select a Bellhop frequency for signal type %s.', p.type);
end
end
