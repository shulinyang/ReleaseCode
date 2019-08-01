function [csi_segment_selected, t, csi_corr, corr_t] = subcarrier_selection(data, n_antennas,n_subcarriers,sample_rate, max_freq, fft_ratio, static_corr)
% [FTP,F,V,T,CSI_CORR,CORR_T]=SPECTRUM_PROCESS(CSI_DATA,SAMPLE_RATE,MAX_FREQ,FFT_RATIO,CARRIER_FREQ,STATIC_CORR)
% calculates spectrogram of CSI data, as in Figure 4a, and correlation
% between subcarriers for motion detection, as in Section 5.1.
%
% DATA     : CSI series (N x 30 subcarriers)
% n_antennas: number of antenna
% n_subcarriers: number of subcarrie
% SAMPLE_RATE  : Sampling rate of CSI data.
% MAX_FREQ     : The upper bound frequency of spectrogram of interest.
% FFT_RATIO    : Time interval length for calculating FFT.
% CARRIER_FREQ : Carrier frequncy of CSI data.
% STATIC_CORR  : CSI correlation in static sceanrios. It is used for
%                subcarrier selection.
%
% csi_segment_selected          : Frequency time profile of CSI data.
% T            : Time axis.
% CSI_CORR     : CSI correlation for motion detection.
% CORR_T       : Timestamp of CSI correlation.
%

fft_window = fft_ratio * sample_rate; % Length of data for FFT.

f = [0:sample_rate/2-1 -sample_rate/2:-1]'/sample_rate; % Frequency axis of FFT.
f_sel = f <= max_freq / sample_rate & f >= 0;
f = f(f_sel,:); % Frequency axis of spectrogram of interest.
corr_n = 20;
for aa = 1:n_antennas
    csi_data=data(:,(aa-1)*n_subcarriers+1:aa*n_subcarriers);
    fft_starts = 1:fft_window/2:size(csi_data, 1) - fft_window + 1; % Start points for FFT.
    ftp = zeros(length(f), (length(fft_starts)+1)*fft_window/2);
    t = ((0:size(ftp, 2)-1)) / sample_rate; % Time axis.
    csi_corr = zeros(1,length(fft_starts));
    csi_segment_selected = zeros(fft_starts(length(fft_starts))+ fft_window - 1,corr_n*n_antennas);
    corr_t = (fft_starts-1)/sample_rate; % Timestamp of CSI correlation.
    for ii = 1:length(fft_starts)
        start_idx = fft_starts(ii);
        csi_segment = csi_data(start_idx:start_idx + fft_window - 1,:);
        
        % Calculaitng CSI correlation
        corr_space = 5; % The space within which correlations are calculated.
        corr_val = zeros(1, size(csi_segment,2));
        for jj = 1:size(csi_segment,2)
            corr_sum = 0;
            corr_cnt = min(size(csi_segment,2), jj+corr_space) - max(1, jj-corr_space);
            magic_number = 10;
            corr_mtx = ones(size(csi_segment,2), size(csi_segment,2)) * magic_number;
            for kk = max(1,jj-corr_space):min(size(csi_segment,2), jj+corr_space)
                if kk == jj
                    continue;
                end
                if corr_mtx(kk, jj) == magic_number
                    corr_curr = corrcoef(csi_segment(:,kk), csi_segment(:,jj));
                    corr_curr = abs(corr_curr(1,2));
                    corr_mtx(kk, jj) = corr_curr;
                    corr_mtx(jj, kk) = corr_curr;
                else
                    corr_curr = corr_mtx(kk, jj);
                end
                corr_sum = corr_curr + corr_sum;
            end
            corr_val(jj) = corr_sum / corr_cnt;
        end
        
        % Subcarrier selection.
        corr_diff = corr_val - static_corr(:,aa).';
        corr_diff_sorted = sort(corr_diff, 'descend');
        csi_sel = corr_diff >= corr_diff_sorted(corr_n);
        
        % CSI correlation for motion detection.
        csi_corr(ii) = mean(corr_val(csi_sel));
        csi_segment_selected(start_idx:start_idx + fft_window - 1,(aa-1)*corr_n+1:aa*corr_n)=csi_segment(:,csi_sel);
    end
end
end
