function unsimiliarity = unsimiliaritycal(xf, yf, sigma, lambda)
    kf = linear_correlation(xf, yf);
    kxf = linear_correlation(xf, xf);
    kyf = linear_correlation(yf, yf);
    yf = single(fft2(gaussian_shaped_labels(0.15 * size(xf, 1), [size(xf, 1), size(xf, 2)])));
    unsimiliarity = exp(-5 * max(max(real(ifft2(yf ./ (kxf + kyf + 2 * lambda) .* (2 * kf))))));
end
