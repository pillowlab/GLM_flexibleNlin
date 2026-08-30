function X = make1Fstimuli(d,nsamps,alpha)
% X = make1Fstimuli(d,nsamps,alpha)
% 
% Draw Gaussian samples with 1/f^alpha power spectrum.
%
% INPUT:
% ------
%      d - dimensionality
% nsamps - number of samples to draw
%  alpha - power for power-law decay of spectrum

% Frequency vector (cycles/sample)
freqs = (0:d-1)';
freqs = min(freqs, d-freqs);  % distance from DC (symmetric frequencies)
freqs(1) = 1;                 % avoid divide-by-zero at DC

% Amplitude spectrum
ampl = freqs.^(-alpha/2);

% Generate white noise in Fourier domain
Xhat = fft(randn(d,nsamps));

% Apply spectral shaping
Xhat = Xhat .* ampl;

% Transform back to real space
X = real(ifft(Xhat));

% Normalize to unit variance
X = X' ./ std(X(:));