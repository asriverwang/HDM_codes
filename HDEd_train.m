function [W, L]=HDEd_train(X, Y, thld, randinit, msg)
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

[N, B]= size(Y{1}); % N: # songs; B: # bins
K = size(X,2);  % K: # acoustic topics
if msg 
    fprintf('HDE training starts with %d topics, %d songs, and %d valence-arousal bin.\n', K, N, B);
    % print the training information
end

% Initialisation
W = cell(1,length(Y));
for d = 1:length(Y)
    if randinit
        Wt = rand(K, B);   
    else
        Wt = ones(K, B);   
    end
    W{d} = Wt./repmat(sum(Wt,2), 1, B);
end


L = [];
r = 0;


for i = 1 : 100 % maximum number of iterations
   
   subL = zeros(length(Y), 1);
   for d = 1:length(Y) 
    
        W{d} = W{d} .* ( (Y{d}./(X*W{d}))' * X)';  % compute the weights  
        W{d}( sum(W{d}, 2)==0, :) = 1; % assign a uniform distribution to the ill-conditioned topic   
        W{d} = W{d} ./ repmat( sum( W{d}, 2), 1, B ); % normalize each topic to sum to one
       
        subL(d) = sum( sum( Y{d} .* log( X*W{d} + eps) ) ); % compute the total log-likelihood
   end
   L_new = sum(subL);
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


