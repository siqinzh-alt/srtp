function samples = make_bpsk_parameter_set(n)


chipRateCandidates = [250, 500, 800, 1000, 1600, 2000];
carrierCandidates = [2200, 2600, 3000, 3400, 3800, 4200, 4600];

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
woaDir = fullfile(root, 'WOA');

for k = 1:n
    dur = 5;
   
    chips_per_sec = chipRateCandidates(randi(numel(chipRateCandidates)));

    pulse = 1.4 + 1.4*rand;              % 1.4--2.8 seconds 
    pulse = round(pulse * chips_per_sec) / chips_per_sec;
    start = 0.4 + (dur-pulse-0.8)*rand;  % leaves quiet margins
    [z, temp, woa] = pick_woa_temperature(woaDir, 100);

    innerDepths = z(2:end-1);
    sd = innerDepths(randi(numel(innerDepths)));%声源
    rd = innerDepths(randi(numel(innerDepths)));%接受

    samples(k).id = sprintf('bpsk_%06d', k);
    samples(k).sig = struct('type', 'BPSK', 'chips_per_sec', chips_per_sec, ...
        'carrier_hz', carrierCandidates(randi(numel(carrierCandidates))), ... 
        'start_time_s', start, 'pulse_width_s', pulse, 'fade_s', 0.02 ...
        );
    samples(k).fs = 16000;
    samples(k).dur = dur;
    samples(k).snr_db = randi([0 15]);
    samples(k).z = z;
    samples(k).temp = temp;
    samples(k).woa = woa;
    samples(k).sd = sd;
    samples(k).rd = rd;
    samples(k).rr = 0.5 + 1.0*rand;      % 水平距离：0.5--1.5 km
end
end
