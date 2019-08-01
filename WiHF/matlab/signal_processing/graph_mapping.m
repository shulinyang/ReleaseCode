function path_matched = graph_mapping(ftp, ftp_list, main_path_group_size)
% path_matched = graph_mapping(ftp, plcr_list, path_sample_rate, sample_rate) 
% extracts path from spectrogram
%
% FTP         : The spectrogram.
% ftp_list   : tha path value candidates.
% path_sample_rate   : Sampling rate of PLCR series.
% SAMPLE_RATE : Sampling rate of spectrogram.
% 
% path_matched        : Path matched.
    group_size = main_path_group_size;
    ftp_dec = zeros(size(ftp,1), ceil(size(ftp,2)/group_size));
    for ii = 1:size(ftp_dec,2)
        if ii < size(ftp_dec,2)
            ftp_dec(:,ii) = mean(ftp(:,(ii-1)*group_size+1:ii*group_size),2);
        else
            ftp_dec(:,ii) = mean(ftp(:,(ii-1)*group_size+1:end),2);
        end
    end

    pow_acc = zeros(size(ftp_dec));
    last_freq = zeros(size(ftp_dec));
    for ii = 1:size(ftp_dec, 2)
        if ii == 1
            pow_acc(:,ii) = ftp_dec(:,ii);
            last_freq(:,ii) = 0;
        else
            for jj = 1:size(ftp_dec, 1)
                if jj == 1
                    [last_pow, idx] = max(pow_acc(jj:jj+1,ii-1));
                    last_freq(jj,ii) = jj+idx-1;
                elseif jj == size(ftp_dec, 1)
                    [last_pow, idx] = max(pow_acc(jj-1:jj,ii-1));
                    last_freq(jj,ii) = jj+idx-2;
                else
                    [last_pow, idx] = max(pow_acc(jj-1:jj+1,ii-1));
                    last_freq(jj,ii) = jj+idx-2;
                end
                pow_acc(jj,ii) = last_pow + ftp_dec(jj,ii);
            end
        end
    end
    
    f = zeros(1, size(ftp_dec,2));
    [~, f(end)] = max(pow_acc(:,end));
    ii = size(pow_acc,2);
    
    while ii > 1
        f(ii-1) = last_freq(f(ii),ii);
        ii = ii - 1;
    end
    path_matched = ftp_list(f);
end