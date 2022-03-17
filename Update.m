function state = Update(state, xf, baglabel)
    t = state.mask;
    x = repmat(t, [length(state.xf) 1]);
    y = repmat(t', [1, length(state.xf)]);
    sm = cellfun(@mynorm, x, y);
    sm = reshape(sm, [length(state.xf) length(state.xf)]);

    for i = 1:length(state.xf)
        sm(i, i) = inf;
    end

    p = repmat({baglabel}, [1, length(state.xf)]);
    s = cellfun(@mynorm, p, t);

    if min(s(:)) <= min(sm(:))
        Case = 1;
    else
        Case = 2;
    end

    switch Case
        case 1
            [~, d] = min(s(:));
            state.xf{d} = state.xf{d} * (1 - state.interp_factor) + xf * state.interp_factor;
            state.mask{d} = (state.mask{d} * state.cs{d} + baglabel) / (state.cs{d} + 1);
            state.cs{d} = state.cs{d} + 1;
        case 2
            [row, col] = find(sm == min(sm(sm > 0)));
            t = col(1);
            k = row(1);
            tmp.xf = state.xf{t};
            tmp.mask = state.mask{t};
            tmp.cs = state.cs{t};
            state.xf{t} = xf;
            state.cs{t} = 1;
            state.mask{t} = baglabel;
            state.xf{k} = state.xf{k} * (1 - state.interp_factor)^tmp.cs + (1 - (1 - state.interp_factor)^tmp.cs) * tmp.cs;
            state.mask{k} = (state.mask{k} * state.cs{k} + tmp.mask * tmp.cs) / (tmp.cs + state.cs{k});
            state.cs{k} = state.cs{k} + tmp.cs;
    end

end

function r = mynorm(x, y)
    r = norm(x - y);
end
