function [W, L]=HDE_train(X, Y, thld, randinit, msg)
%
% INPUT PARAMETERS:
%   Y (label): song-by-bin matrix (each row allocates the histogram density of a song)
%	X (feature): song-by-(acoustic)topic matrix, containing entries (each row allocates the acoustic posterior of a song)
%   thld: stopping threshold for the log-likelihood increase ratio
%   randinit: {0, 1} indicates using random initialization for the parameters or not
%   msg: {0, 1} indicates showing the training messages or not
% OUTPUTS:
%	W (model): (acoustic)topic-by-bin matrix (each row represents a topic) 
%   L: records the log-likelihood per iteration


if nargin<3
    % default setting
    thld = 0.0001;
    randinit = 0;
    msg  = 0;
end

[N, B]= size(Y); % N: # songs; B: # bins
K = size(X,2);  % K: # acoustic topics
if msg 
    fprintf('HDE training starts with %d topics, %d songs, and %d valence-arousal bin.\n', K, N, B);
    % print the training information
end

% Initialisation
if randinit
    W = rand(K, B);   
else
    W = ones(K, B);   
end

W = W./repmat(sum(W,2), 1, B);
L = [];
r = 0;


for i = 1 : 100 % maximum number of iterations

   W = W .* ( (Y./(X*W))' * X)';  % compute the weights
   
   W( sum(W,2)==0, :) = 1; % assign a uniform distribution to the ill-conditioned topic
   
   W = W ./ repmat( sum( W, 2), 1, B ); % normalize each topic to sum to one
   
   
   L_new = sum( sum( Y .* log( X*W + eps) ) ); % compute the total log-likelihood
   L = [L, L_new]; % record the log-likelihood per iteration
   
   if (i>1)
       r = (L(i)-L(i-1))/abs(L(i-1)); % compute the increase ratio of log-likelihood
   end
   
   if msg
       fprintf('iteration %3d, L = %f, ratio = %.6f \n', i, L(i), r);
       % print the training information
   end

   if (r<thld && i>1)  % stopping criterion
      if msg
           fprintf('training stops at iteration %3d.\n', i);
           % print the training information
      end
      break;
   end   
end


