function bb = build_basis_polar_mat(angs,chan_centers,chan_width,chan_pow )
% adapted from build_basis_polar - this does everything at once, useful for
% generating a set of transformed/rotated basis functions

% will return bb, length(evalAt) x length(rfTh)
n_basis = length(chan_centers);
[chan_centers,angs] = meshgrid(squeeze(chan_centers),squeeze(angs));

if nargin < 3 || isempty(chan_width)
    chan_width = 180; % default value
end

if nargin < 4 || isempty(chan_pow)
    chan_pow = n_basis-mod(n_basis,2);
end

% utility function to compute distance between two angles
ang_dist = @(a,b) min(mod(a-b,180),mod(b-a,180));


bb = (cosd( 180 * ang_dist(angs,chan_centers) ./ (2*chan_width) ).^chan_pow) .* (ang_dist(angs,chan_centers)<=chan_width) ;


return