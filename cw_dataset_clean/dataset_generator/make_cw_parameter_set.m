function samples = make_cw_parameter_set(n)
%MAKE_CW_PARAMETER_SET Sample diverse, physically valid CW conditions.
%   The random generator is set by the caller.  Every sampled value is
%   retained in the per-sample JSON and manifest, so the dataset remains
%   reproducible and auditable.

arguments
    n (1,1) double {mustBeInteger, mustBePositive}
end

z = [0; 10; 20; 40; 60; 80; 100];
for k = 1:n
    dur = 5;
    pulse = 1.4 + 1.4*rand;              % 1.4--2.8 seconds
    start = 0.4 + (dur-pulse-0.8)*rand;  % leaves quiet margins

    surfaceTemp = 22 + 4*rand;
    bottomTemp = 13.5 + 3.0*rand;
    temp = linspace(surfaceTemp, bottomTemp, numel(z)).';
    salt = linspace(33.8 + 0.4*rand, 34.6 + 0.35*rand, numel(z)).';

    innerDepths = z(2:end-1);
    sd = innerDepths(randi(numel(innerDepths)));
    rd = innerDepths(randi(numel(innerDepths)));

    samples(k).id = sprintf('cw_%06d', k);
    samples(k).sig = struct('type', 'CW', ...
        'carrier_hz', round(1800 + 3600*rand), ... % 1.8--5.4 kHz
        'start_time_s', start, 'pulse_width_s', pulse, 'fade_s', 0.02);
    samples(k).fs = 16000;
    samples(k).dur = dur;
    samples(k).snr_db = randi([0 15]);
    samples(k).z = z;
    samples(k).temp = temp;
    samples(k).salt = salt;
    samples(k).sd = sd;
    samples(k).rd = rd;
    samples(k).rr = 0.5 + 1.0*rand;      % 0.5--1.5 km
end
end
