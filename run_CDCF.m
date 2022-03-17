function res = run_CDCF(video_path, seqname, param)
    disp(['Tracking ' seqname]);
    res = [];
    img_file = dir([video_path '\' seqname '\img\*.jpg']);
    img_file = fullfile([video_path '\' seqname '\img'], {img_file.name});
    subS.s_frames = img_file;
    %%load groundtruth_rect
    if exist([video_path '\' seqname '\cfg.mat'], 'file')
        a = load([video_path '\' seqname '\cfg.mat']);

        if isfield(a, 'seq')
            ground_truth = a.seq.gt_rect;
        else
            ground_truth = a.groundtruthrect;
        end

    else

        try
            ground_truth = load([video_path '\' seqname '\groundtruth_rect.txt']);
        catch

            ground_truth = load([video_path '\' seqname '\groundtruth_rect.1.txt']);
        end

        if ~exist('ground_truth', 'var')
            disp(['openfilefailed:' video_path '\' seqname '\groundtruth_rect.txt' '||' video_path '\' seqname '\groundtruth_rect.1.txt']);
            return;
        end

    end

    init_rect = ground_truth(1, :);
    subS.init_rect = init_rect; %1-index
    im = vl_imreadjpeg(subS.s_frames, 'numThreads', 12);
    num_frame = numel(im);
    result = repmat(init_rect, [num_frame, 1]);
    init_rect(1:2) = init_rect(1:2) - 1; %0-index
    [state, ~] = CDCF_initialize(im{1}, init_rect, param);
    tic;

    for frame = 2:num_frame
        [state, region] = CDCF_update(state, im{frame});
        region(1:2) = region(1:2) + 1; %1-index
        result(frame, :) = region;
    end

    time = toc;
    res.type = 'rect';
    res.res = result;
    res.fps = num_frame / time;
    res.num_frame = num_frame;
    save(['result\' seqname '.mat'], 'result');
    disp([seqname ' Finished']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%member function
function [state, location] = CDCF_initialize(I, region, param)
    state.rateno = 1;
    state.gpu = true;
    state.visual = false;
    state.match = [];
    state.bagpos = [];
    state.psz = 25;
    state.nsz = 24;
    state.beta = {8/15 4/15 2/15 1/15};
    state.bagresth = 0.3;
    state.lambda = 1e-4;
    state.padding = 1;
    state.output_sigma_factor = 0.1;
    state.interp_factor = 0.0105;
    state.bagsigma = 10;
    state.num_scale = 3;
    state.scale_step = 1.015;
    state.min_scale_factor = 0.2;
    state.max_scale_factor = 5;
    state.scale_penalty = 1;
    state.net = [];
    state = vl_argparse(state, param);
    state.scale_factor = state.scale_step.^((1:state.num_scale) - ceil(state.num_scale / 2));
    state.scale_penalties = ones(1, state.num_scale);
    state.scale_penalties((1:state.num_scale) ~= ceil(state.num_scale / 2)) = state.scale_penalty;
    state.net_input_size = state.net.meta.normalization.imageSize(1:2);
    state.net_average_image = state.net.meta.normalization.averageImage;
    state.output_sigma = sqrt(prod(state.net_input_size ./ (1 + state.padding))) * state.output_sigma_factor;
    state.yf = single(fft2(gaussian_shaped_labels(state.output_sigma, state.net_input_size)));
    state.cos_window = single(hann(size(state.yf, 1)) * hann(size(state.yf, 2))');
    yi = linspace(-1, 1, state.net_input_size(1));
    xi = linspace(-1, 1, state.net_input_size(2));
    [xx, yy] = meshgrid(xi, yi);
    state.yyxx = single([yy(:), xx(:)]'); % yyxx=2xM matrix

    if state.gpu %gpuSupport
        state.yyxx = gpuArray(state.yyxx);
        state.net = vl_simplenn_move(state.net, 'gpu');
        I = gpuArray(I);
        state.yf = gpuArray(state.yf);
        state.cos_window = gpuArray(state.cos_window);
    end

    state.pos = region([2, 1]) + region([4, 3]) / 2;
    state.target_sz = region([4, 3])';
    state.min_sz = max(4, state.min_scale_factor .* state.target_sz);
    [im_h, im_w, ~] = size(I);
    state.max_sz = min([im_h; im_w], state.max_scale_factor .* state.target_sz); %ç›®æ �?çš„æœ€å¤§å°ºå¯¸
    window_sz = state.target_sz * (1 + state.padding);
    patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
    [fm] = featureExt(patch, state);
    x = bsxfun(@times, fm, state.cos_window); %x=125x125x32
    xf = fft2(x);
    state.numel_xf = numel(xf);
    state.xf = repmat({xf}, [1 length(state.beta)]);
    state.cs = repmat({1}, [1 length(state.beta)]);
    init_label = -1 + zeros(7, 7);
    init_label(2:6, 2:6) = init_label(2:6, 2:6) + 2;
    state.mask = repmat({init_label}, [1 length(state.beta)]);
    location = region;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%sampling here%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [row, col, ~] = size(fm);
    [xymin] = round(0.5 * state.padding / (1 + state.padding) * [row col])' + 1;
    [xymax] = round((1 - 0.5 * state.padding / (1 + state.padding)) * [row col])'; %èŽ·å��?å·¦ä¸Šç‚¹å�?Œå�³ä¸�?ç‚�?
    state.bagsz = round(0.8 * [min(xymax - xymin + 1); min(xymax - xymin + 1)]); %bagå°ºå¯¸ä¸º0.5
    state.bagstep = round(0.5 * state.bagsz);
    [ppool, npool, plabel, nlabel] = sampling(fm, state, xymin, xymax); %æ­£è´Ÿæ ·æœ¬é‡�?æ ·
    [pmodel, nmodel] = createmodel(ppool, npool, state); %ç”Ÿæˆ�æ¨¡åž�?
    state.pmodel = pmodel;
    state.nmodel = nmodel;
    state.plabel = plabel;
    state.nlabel = nlabel;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [state.pmodel, state.ppsm, state.plabel] = calculatesm(state.pmodel, state.psz, state.plabel, state);
    [state.nmodel, state.nnsm, state.nlabel] = calculatesm(state.nmodel, state.nsz, state.nlabel, state);

    if state.visual
        state.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(I, 2), size(I, 1)] + 30]);
    end

end

function [pmodel, nmodel] = createmodel(ppool, npool, state)
    pmodel = [];
    nmodel = [];
    coswin = ones(size(ppool{1}, 1), size(ppool{1}, 2));
    yf = single(fft2(gaussian_shaped_labels(0.15 * size(ppool{1}, 1), [size(ppool{1}, 1) size(ppool{1}, 2)])));
    lambda = 1e-4;

    for i = 1:length(ppool)
        x = ppool{i};
        x = bsxfun(@times, x, coswin);
        xf = fft2(x);
        numel_xf = numel(xf);
        kf = gaussian_correlation(xf, xf, state.bagsigma);
        model.alphaf = yf ./ (kf + lambda);
        model.xf = xf;
        model.w = 1;
        pmodel{end + 1} = model;
    end

    for i = 1:length(npool)
        x = npool{i};
        x = bsxfun(@times, x, coswin);
        xf = fft2(x);
        numel_xf = numel(xf);
        kf = gaussian_correlation(xf, xf, state.bagsigma);
        model.alphaf = yf ./ (kf +lambda);
        model.xf = xf;
        model.w = 1;
        nmodel{end + 1} = model;
    end

end

function [state, location] = CDCF_update(state, I, varargin)
    state.rateno = state.rateno + 1;
    if state.gpu, I = gpuArray(I); end
    window_sz = bsxfun(@times, state.target_sz, state.scale_factor) * (1 + state.padding);
    patch_crop = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
    [fm] = featureExt(patch_crop, state);
    z = bsxfun(@times, fm, state.cos_window);
    zf = fft2(z);
    [~, vert_delta, horiz_delta, scale_delta] = Detect(zf, state);
    window_sz = window_sz(:, scale_delta);
    state.pos = state.pos + [vert_delta - 1, horiz_delta - 1] .* ...
        window_sz' ./ state.net_input_size;
    state.target_sz = min(max(window_sz ./ (1 + state.padding), state.min_sz), state.max_sz);

    patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
    %%%%%%%%%%%%%%%%%%%%%%%%bag search here%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [bagres, up] = bagsearch(patch, state);
    state.match{end + 1} = sum(bagres.baglabel(:) == 1);
    bagres.pos = bagres.pos ./ window_sz ./ state.net_input_size';
    state.bagpos = cat(2, state.bagpos, gather(bagres.pos));
    %%%%%%%%%%%%%%%%%%%%%%%%%%bag model update%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(state.rateno, 10) == 0
        [state.pmodel, state.ppsm, state.plabel] = bagmodelupdate(state.pmodel, up.pup, state.psz, state.ppsm, state.bagsigma, state.plabel, state.lambda);
        [state.nmodel, state.nnsm, state.nlabel] = bagmodelupdate(state.nmodel, up.nup, state.nsz, state.nnsm, state.bagsigma, state.nlabel, state.lambda);
    end

    [fm] = featureExt(patch, state);
    x = bsxfun(@times, fm, state.cos_window);
    xf = fft2(x);
    state = Update(state, xf, bagres.baglabel);
    box = [state.pos([2, 1]) - state.target_sz([2, 1])' / 2, state.target_sz([2, 1])'];
    testbox = [gather(state.pos([2, 1]) + bagres.pos([2, 1])') - state.target_sz([2, 1])' / 2, state.target_sz([2, 1])'];
    location = double(gather(testbox));

    if state.visual
        testbox = [gather(state.pos([2, 1]) + bagres.pos([2, 1])') - state.target_sz([2, 1])' / 2, state.target_sz([2, 1])'];
        % im_show = insertShape(uint8(gather(I)), 'Rectangle', location, 'LineWidth', 4, 'Color', 'yellow');
        im_show = insertShape(uint8(gather(I)), 'Rectangle', testbox, 'LineWidth', 4, 'Color', 'green');

        step(state.videoPlayer, im_show);
    end

    state.pos = state.pos + bagres.pos';
end

function [bagsf, baglabel] = SearchSampleAndFFT(fm, state)
    coswin = ones(state.bagsz(1), state.bagsz(1));
    sz = size(fm);
    num = (sz(1:2)' ./ state.bagsz);
    bagsf = [];
    baglabel = zeros(length(1:0.25:num(1)), length(1:0.25:num(2)));

    for i = 1:0.25:num(1)
        x = round(0 + (i - 1) * state.bagsz(1));

        for j = 1:0.25:num(2)
            y = round(0 + (j - 1) * state.bagsz(2));
            bag = fm(x + 1:x + state.bagsz(1), y + 1:y + state.bagsz(2), :);
            bag = bsxfun(@times, bag, coswin);
            bagsf{end + 1} = fft2(bag);
        end

    end

end

function [bagres, up] = bagsearch(patch, state)
    [fm] = featureExt(patch, state);
    [bagsf, baglabel] = SearchSampleAndFFT(fm, state);
    %%%%%%%%%%%%%%%%%bag search%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nb = length(bagsf);
    np = length(state.pmodel);
    nn = length(state.nmodel);
    sz = [size(bagsf{1}, 1) size(bagsf{1}, 2)];
    model = [state.pmodel, state.nmodel];
    b = repmat(bagsf', [1 np + nn]);
    model = repmat(model, [nb, 1]);
    [res, d] = cellfun(@responseforcell, b, model, repmat(num2cell(state.bagsigma), [nb, np + nn]), repmat(mat2cell(sz, [1], [2]), [nb, np + nn]), 'UniformOutput', false);

    if state.gpu
        res = gpuArray(cell2mat(res));
        d = gpuArray(cell2mat(d));
    else
        res = cell2mat(res);
        d = cell2mat(d);
    end

    [max_response, max_index] = max(res');
    bagres = [];
    up = [];
    pup = [];
    nup = [];
    res_sum = 1e-8;
    baglocate = zeros(2, nb);
    max_response = max_response - state.bagresth;

    for i = 1:length(max_response)

        if max_response(i) > 0

            if max_index(i) <= np %åŒ¹é…�
                baglabel(fix((i - 1) / size(baglabel, 1)) + 1, mod(i - 1, size(baglabel, 2)) + 1) = 1;

                baglocate(:, i) =- [d(i, max_index(i), 1); d(i, max_index(i), 2)] + state.plabel{max_index(i)} + 1;
                t = 0.25 * state.bagsz .* ([2 + fix((i - 1) / 5); 2 + mod(i - 1, 5)]) - 0.5 * 125;
                res_sum = res_sum + max_response(i); %å½’ä¸€åŒ�?
                tmp.loc = t;
                tmp.w = 0.01 * (max_response(i) - state.bagresth);
                tmp.xf = bagsf{i};
                pup{end + 1} = tmp;
            else %é�®æŒ�?
                baglabel(fix((i - 1) / size(baglabel, 1)) + 1, mod(i - 1, size(baglabel, 2)) + 1) = -1;
                t = 0.25 * state.bagsz .* ([2 + fix((i - 1) / 5); 2 + mod(i - 1, 5)]) - 0.5 * 125;
                tmp.loc = t;
                tmp.w = 0.01 * (max_response(i) - state.bagresth);
                tmp.xf = bagsf{i};
                nup{end + 1} = tmp;
            end

        end

    end

    up.pup = pup;
    up.nup = nup;
    patsz = zeros(2, 1);
    patsz(1) = size(patch, 1);
    patsz(2) = size(patch, 2);
    pos = [0; 0];

    for i = 1:size(baglabel, 1)

        for j = 1:size(baglabel, 2)

            if baglabel(i, j) == 1
                pos = pos + (0.5 * state.bagsz + [(i - 1); (j - 1)] .* state.bagsz * 0.25 - 0.5 * patsz - baglocate(:, (i - 1) * size(baglabel, 2) + j)) * max_response((i - 1) * size(baglabel, 2) + j) / res_sum;
            end

        end

    end

    bagres.baglabel = baglabel;
    bagres.pos = pos;
end

function img_crop = imcrop_multiscale(img, pos, sz, output_sz, yyxx)
    [im_h, im_w, im_c, ~] = size(img);

    if im_c == 1
        img = repmat(img, [1, 1, 3, 1]);
    end

    cy_t = (pos(1) * 2 / (im_h - 1)) - 1;
    cx_t = (pos(2) * 2 / (im_w - 1)) - 1; %æ˜ å°„åˆ°[-1,1]
    h_s = sz(1, :) / (im_h - 1);
    w_s = sz(2, :) / (im_w - 1);
    s = reshape([h_s; w_s], 2, 1, []); % x,y scaling
    t = [cy_t; cx_t]; % translation
    g = bsxfun(@times, yyxx, s); % scale
    g = bsxfun(@plus, g, t); % translate
    g = reshape(g, 2, output_sz(1), output_sz(2), []);
    img_crop = vl_nnbilinearsampler(img, g);
end
