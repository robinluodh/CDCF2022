function [scale] = ScaleEstimate(response)
    [row, col, dep] = size(response);
    res = mat2cell(response, row, col, ones(1, dep));
    score = cellfun(@APCEest, res);
    [~, scale] = max(score);
end

function s = APCEest(res)
    [row, col] = size(res);
    maxres = max(res(:));
    minres = min(res(:));
    tmp = (res - minres).^2;
    tmp = sum(tmp(:)) / col / row;
    s = (maxres - minres)^2 / tmp;
end
