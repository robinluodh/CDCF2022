function [fm] = featureExt(I, state)
    search = bsxfun(@minus, I, state.net_average_image);
    res = vl_simplenn(state.net, search);
    fm = res(end).x;
end
