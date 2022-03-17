function [response, vert_delta, horiz_delta, scale_delta, confidence] = Detect(zf, state)
    %DETECT Do Fast Detect
    %   DCF filter
    cs = cell2mat(state.cs);
    [~, ind] = sort(cs, 'descend');
    kf = cellfun(@linear_correlation, state.xf, state.xf, 'UniformOutput', false);
    kf = cellfun(@times, state.beta, kf(ind), 'UniformOutput', false);
    d = state.lambda + sum(cat(3, kf{:}), 3);
    zf = repmat({zf}, [1, length(state.xf)]);
    kzf = cellfun(@linear_correlation, zf, state.xf, 'UniformOutput', false);
    kzf = cellfun(@times, state.beta, kzf(ind), 'UniformOutput', false);
    kzf = sum(squeeze(cat(3, kzf{:})), 3);
    n = state.yf;
    alphaf = n ./ d;
    response = squeeze(real(ifft2(bsxfun(@times, alphaf, kzf)))); %response 125x125x3
    [max_response, max_index] = max(reshape(response, [], state.num_scale));
    max_response = gather(max_response);
    max_index = gather(max_index);
    [confidence, scale_delta] = max(max_response .* state.scale_penalties);
    %scale_delta=ScaleEstimate(response);
    [vert_delta, horiz_delta] = ind2sub(state.net_input_size, max_index(scale_delta));

    if vert_delta > state.net_input_size(1) / 2 %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - state.net_input_size(1);
    end

    if horiz_delta > state.net_input_size(2) / 2 %same for horizontal axis
        horiz_delta = horiz_delta - state.net_input_size(2);
    end

end
