function [res, d] = responseforcell(b, m, bagsigma, sz)
    bagsf = b;
    alphaf = m.alphaf;
    xf = m.xf;
    kzf = gaussian_correlation(bagsf, xf, bagsigma);
    response = real(ifft2(bsxfun(@times, alphaf, kzf)));
    [maxres, max_index] = max(response(:));
    [vert_delta, horiz_delta] = ind2sub(sz, max_index);

    if vert_delta > sz(1) / 2 %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - sz(1);
    end

    if horiz_delta > sz(2) / 2 %same for horizontal axis
        horiz_delta = horiz_delta - sz(2);
    end

    res = gather(maxres);
    d = gather(cat(3, vert_delta, horiz_delta));
end
