function plotSig(f, x, H, height, color, width)
% h: handle for figure
% x: xaxis
% H: sig value (0 or 1, 1 for significant)

if nargin < 5
    color = 'k';
end

if nargin < 6
    width = 0.75;
end

% turn H to row vector
if(size(H, 1)~=1)
   H = H'; 
end

% find start_x and end_x for sig segment
dH = diff([0,H,0]);
[~, start_x] = find(dH == 1);
[~, end_x] = find(dH == -1);
end_x = end_x - 1;
% plot
figure(f);
hold on;

n_points = length(start_x);
y = [height, height];
for i = 1:n_points
    plot([x(start_x(i)), x(end_x(i))], y, ...
        'Color', color, 'LineWidth', width);
end
    
end

