function [p, ppsm, label] = calculatesm(p, sz, label, state)
    %this function calculates the similiarity matrix
    %ppsm saves the distance/unsimiliarity between samples in the pool
    n = length(p);
    ppsm = zeros(sz, sz);

    if n <= sz

        for i = 1:n

            for j = 1:n

                ppsm(i, j) = unsimiliaritycal(p{i}.xf, p{j}.xf, state.bagsigma, state.lambda);
            end

        end

    else
        t = [];

        for i = 1:sz
            t{end + 1} = p{i};

            for j = 1:sz

                ppsm(i, j) = unsimiliaritycal(p{i}.xf, p{j}.xf, state.bagsigma, state.lambda);
            end

        end

        tmp = p;
        p = t;
        tmp(1:sz) = [];

        for k = 1:length(tmp)
            tmp{k}.loc = label{k + sz};
            tmp{k}.w = 1;
        end

        [p, ppsm, label] = bagmodelupdate(p, tmp, sz, ppsm, state.bagsigma, state.plabel, state.lambda);
    end

end
