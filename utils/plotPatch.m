function plotPatch(h, x, m, sd, color, alpha)
% h: handle for figure
% x: xaxis, row vector
% m: mean for data, row vector
% sd: std for data, row vector

if nargin < 6
    alpha = 0.3;
end

% generate points x - y(m) for patch
x = [x, fliplr(x)];
m = [m+sd, fliplr(m-sd)];

patch(h, x, m, color, 'FaceAlpha',alpha, 'LineStyle', 'none');

end

