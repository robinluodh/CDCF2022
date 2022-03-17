function [ppool, npool, plabel, nlabel] = sampling(featuremap, state, xymin, xymax)
    %sample - Description
    %
    %
    %
    % make postive and negative samples here
    %%
    sz = [size(featuremap, 1); size(featuremap, 2)];
    xl = [0:0.25:(sz(1) - state.bagsz(1)) / state.bagsz(1)];
    yl = [0:0.25:(sz(2) - state.bagsz(2)) / state.bagsz(2)];

    %postive:
    ppool = [];
    plabel = [];
    nlabel = [];
    npool = [];
    totalpool = [];
    center = round(0.5 * (xymin + xymax) - 0.5 * state.bagsz);

    for i = 1:length(xl)
        linpool = [];
        xmin = 1 + fix(xl(i) * state.bagsz(1));
        xmax = 1 + fix(xl(i) * state.bagsz(1)) + state.bagsz(1) - 1;

        if 0.5 * (xmin + xmax) >= xymin(1) && 0.5 * (xmin + xmax) <= xymax(1)
            xin = 1;
        else
            xin = 0;
        end

        for j = 1:length(yl)
            %ppool=cat(4,ppool,featuremap(xymin(1)+(i-1)*state.bagstep:xymin(1)+(i-1)*state.bagstep+state.bagsz,xymin(2)+(j-1)*state.bagstep:xymin(2)+(j-1)*state.bagstep+state.bagsz,:));
            %ppool{end+1}=featuremap(xymin(1)+(i-1)*state.bagstep:xymin(1)+(i-1)*state.bagstep+state.bagsz,xymin(2)+(j-1)*state.bagstep:xymin(2)+(j-1)*state.bagstep+state.bagsz,:);
            ymin = 1 + fix(yl(j) * state.bagsz(2));
            ymax = 1 + fix(yl(j) * state.bagsz(2)) + state.bagsz(2) - 1;

            if 0.5 * (ymin + ymax) >= xymin(2) && 0.5 * (ymin + ymax) <= xymax(2)
                yin = 1;
            else
                yin = 0;
            end

            if xin && yin
                ppool{end + 1} = featuremap(xmin:xmax, ymin:ymax, :);
                plabel{end + 1} = -center + [xmin; ymin];
            else
                npool{end + 1} = featuremap(xmin:xmax, ymin:ymax, :);
                nlabel{end + 1} = -center + [xmin; ymin];
            end

            tmp = cat(2, featuremap(xmin:xmax, ymin:ymax, :), xin * yin * 255 * ones(xmax - xmin + 1, 1, size(featuremap, 3)));
            tmp = cat(1, tmp, xin * yin * 255 * ones(1, ymax - ymin + 2, size(featuremap, 3)));
            linpool = cat(2, linpool, tmp);
        end

        totalpool = cat(1, totalpool, linpool);
    end

    %save('debag.mat','ppool','npool','totalpool','-append');%for debag

end
