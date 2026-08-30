function L = mkPrecision_graphLapl(nw,DCflag)
% L = mkPrecision_graphLapl(nw,DCflag)
%
% Build precision (inverse covariance matrix) corresonding to a Graph
% Laplacian regularization penalty 
%
% Inputs:
% -------
%        nw - length of param vector to apply to prior to
%    DCflag - boolean for whether last element is a DC offset (0 = not DC; 1 = is DC)
%
% Outputs:
% --------
%   L [nw x nw] or [nw+1 x nw+1 ] - inverse covariance matrix 
%
% Notes:
% ------
%
% Graph laplacian is given by:
%
%  L = [ 1 -1
%             -1 2 -1                       
%                .   .   .
%                  -1 2 -1
%                    -1  1 ]


if nargin == 1
    DCflag = 0;   % assume no DC dterm
end

vdiag = [1;2*ones(nw-2,1); 1]; % diagonal values
voffdiag = -ones(nw,1);  % off-diagonal values
L = spdiags([voffdiag,vdiag,voffdiag],-1:1,nw,nw);

if DCflag  
    % add a row and a column of all zeros, for DC term
    L = blkdiag(L,0);
end
