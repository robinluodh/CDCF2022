%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%padding%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Before running ,make sure that you have compiled the matconvnet toolkit.
%An compiling command can be :   vl_compilenn('enableGpu', true, 'cudaRoot', 'D:\CUDA11.4','cudaMethod' ,'nvcc','verbose', '2')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo(mode, bvisual)
    %edit your sequences path here:
    video_path = ['E:\OTB\OTB100'];

    if nargin >= 1 && strcmp(mode, 'all')
    else
        mode = 'switch';
    end

    if nargin >= 2 && bvisual == false
    else
        bvisual = true;
    end

    %%init workspace
    str = strsplit(pwd, '\');

    if strcmp(str{end}, 'go_from_here')
        cd '..\'
    end

    addpath(genpath('.\'));
    %%load net
    net_name = 'DCFnet.mat';
    net = load(net_name);
    net = vl_simplenn_tidy(net.net);
    %%init parameter
    param = [];
    param.gpu = false;
    param.visual = bvisual;
    param.net = net;
    param.padding = 1;
    %%choose sequences
    if strcmp(mode, 'switch')
        seqname = choose_video(video_path);
        res = run_CDCF(video_path, seqname, param);
    else
        seq = dir(video_path);
        [num, ~] = size(seq);
        size(seq);
        existseq = dir('result\');

        if num >= 3

            parfor i = 3:num

                if seq(i).isdir
                    res = run_CDCF(video_path, seq(i).name, param);
                    %%add to do with result here
                    %TO DO:
                end

            end

        else
            error('NO video in sequences');
        end

    end

end
