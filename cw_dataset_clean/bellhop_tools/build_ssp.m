function [SSP, Bdry, Pos, Beam, cInt, RMax, c] = build_ssp(z, temp, salt, sd, rd, rr)
%BUILD_SSP Build all structures required by Bellhop write_env.m.

z = z(:); temp = temp(:); salt = salt(:);
assert(numel(z) >= 2 && numel(z) == numel(temp) && numel(z) == numel(salt), ...
    'z, temp, and salt must have equal lengths.');
assert(all(diff(z) > 0) && z(1) == 0, ...
    'z must start at 0 m and increase strictly with depth.');
assert(sd >= 0 && sd <= z(end) && rd >= 0 && rd <= z(end), ...
    'Source and receiver depths must be inside the SSP water column.');

% Mackenzie-style empirical sound-speed equation (m/s).
c = 1449.2 + 4.6*temp - 0.055*temp.^2 + 0.00029*temp.^3 + ...
    (1.34 - 0.01*temp).*(salt - 35) + 0.016*z;

% The field names below are fixed by Bellhop write_env.m.
SSP.NMedia = 1;
SSP.N = numel(z);
SSP.sigma = 0;
SSP.depth = [0; z(end)];
SSP.raw(1).z = z;
SSP.raw(1).alphaR = c;
SSP.raw(1).betaR = zeros(size(z));
SSP.raw(1).rho = ones(size(z));
SSP.raw(1).alphaI = zeros(size(z));
SSP.raw(1).betaI = zeros(size(z));

Bdry.Top.Opt = 'SVW';
Bdry.Top.depth = 0;
Bdry.Bot.Opt = 'A';
Bdry.Bot.depth = z(end);
Bdry.Bot.HS.alphaR = 1600;
Bdry.Bot.HS.betaR = 0;
Bdry.Bot.HS.rho = 1.8;
Bdry.Bot.HS.alphaI = 0;
Bdry.Bot.HS.betaI = 0;

Pos.s.depth = sd;
Pos.r.depth = rd;
Pos.r.range = rr;

Beam.RunType = 'A';
Beam.Nbeams = 101;
Beam.alpha = [-30 30];
Beam.deltas = 0;
Beam.Box.z = z(end) + 10;
Beam.Box.r = rr + 0.2;

cInt.Low = min(c) - 20;
cInt.High = max(c) + 20;
RMax = Beam.Box.r;
end
