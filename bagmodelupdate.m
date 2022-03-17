function [p, sm, label] = bagmodelupdate(pool, up, psz, ppsm, sigma, l, lambda)
    %this function updates the positive and negative patch filter(in code,called model);
    %pool is the existing positive model pool

    %up is a set of matched patches, with their confidence/weights(the response);
    %the update strategy consist of 4 cases:
    %case 1: pool is not full
    %           in this situation,just add it to the pool
    %case 2: pool full,out the least visited/used model
    %           the least visited/used indicates least importance/weight
    %case 3: pool full,merge the new one
    %           merge it to a existing model,and updates the weight
    %case 4: pool full,the new one varies from others,merge 2 existing models
    %           the new one may contain new information,so choose to keep it and merge 2 existing models,of course update the weight
    %
    label = l;
    sm = ppsm;
    p = pool;
    yf = single(fft2(gaussian_shaped_labels(0.15 * size(pool{1}.xf, 1), [size(pool{1}.xf, 1) size(pool{1}.xf, 2)])));

    for j = 1:length(up)
        loc = up{j}.loc;
        model.xf = up{j}.xf;
        kf = gaussian_correlation(model.xf, model.xf, sigma);
        model.w = up{j}.w;
        model.alphaf = yf ./ (kf + lambda);

        if length(p) < psz
            Case = 1;
        else
            w = zeros(1, psz);
            d = zeros(1, psz);
            dk = zeros(1, psz);
            dt = zeros(1, psz);

            for i = 1:psz
                w(i) = p{i}.w;
                d(i) = unsimiliaritycal(model.xf, p{i}.xf, sigma, lambda);
            end

            %     if min(d(:))<=min(sm(sm>0.0067))
            %         Case=3;
            %
            %     else
            %
            %          if     max(d(:))>=max(sm(sm~=inf))
            %             Case=4;
            %
            %          else
            %              Case=2;
            %          end
            % %     end
            %     if min(d)<=min(sm(sm>0.0067))
            %         Case=3;
            %     else
            %         Case=4;
            %     end
            Case = 3;
        end

        switch Case
            case 1

                for i = 1:length(p)
                    sm(i, length(p) + 1) = unsimiliaritycal(model.xf, p{i}.xf, sigma, lambda);
                    sm(length(p) + 1, i) = sm(i, length(p) + 1);
                end

                sm(length(p) + 1, length(p) + 1) = 1e-2;
                p{end + 1} = model;
                label{end + 1} = loc;

            case 2
                [~, col] = find(w == min(w));

                col = col(randperm(numel(col)));

                p{col(1)} = model;
                label{col(1)} = loc;
                d(1, col(1)) = 1e-2;

                sm(:, col(1)) = d';

                sm(col(1), :) = d;
            case 3
                k = find(d == min(d));
                k = k(1);
                label{k} = (label{k} * p{k}.w + loc * model.w) / (p{k}.w + model.w);
                p{k}.xf = (p{k}.xf * p{k}.w + model.xf * model.w) / (p{k}.w + model.w);
                p{k}.alphaf = (p{k}.alphaf * p{k}.w + model.alphaf * model.w) / (p{k}.w + model.w);
                p{k}.w = (p{k}.w + model.w);

                for i = 1:psz
                    d(i) = unsimiliaritycal(p{k}.xf, p{i}.xf, sigma, lambda);
                end

                sm(:, k) = d';
                sm(k, :) = d;
            case 4
                [row, col] = find(sm == min(sm(sm > 0.0067)));
                t = row(1);
                tm = p{t};
                tl = label{t};
                k = col(1);
                p{t} = model;
                label{t} = loc;
                label{k} = (label{k} * p{k}.w + tl * tm.w) / (p{k}.w + tm.w);
                p{k}.xf = (p{k}.xf * p{k}.w + tm.xf * tm.w) / (p{k}.w + tm.w);
                p{k}.alphaf = (p{k}.alphaf * p{k}.w + tm.alphaf * tm.w) / (p{k}.w + tm.w);
                p{k}.w = (p{k}.w + tm.w);

                for i = 1:psz
                    dk(i) = unsimiliaritycal(p{k}.xf, p{i}.xf, sigma, lambda);
                    dt(i) = unsimiliaritycal(p{t}.xf, p{i}.xf, sigma, lambda);
                end

                dk(k) = 1e-2;
                dt(t) = 1e-2;
                sm(:, k) = dk';
                sm(k, :) = dk;
                sm(:, t) = dt';
                sm(t, :) = dt;
        end

    end

end
