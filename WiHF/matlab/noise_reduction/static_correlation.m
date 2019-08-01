function corr_val = static_correlation(csi_data, n_antennas, n_subcarriers, sample_rate, start_time, duration)
% CORR_VAL=STATIC_CORRELATION(CSI_DATA, N_DEVICES, N_ANTENNAS,
% N_SUBCARRIERS, SAMPLE_RATE, MAX_FREQ, MIN_FREQ, START_TIME, DURATION)
% calculates the correlation of CSI subcarriers in static scenarios.
%
% CSI_DATA      : Raw CSI measurements, eg:dim($(time_series),n_antennas*n_subcarriers)
% N_ANTENNAS    : Number of antennas per devices
% N_SUBCARRIERS : Number of subcarriers in CSI
% SAMPLE_RATE   : Target sampling rate of interpolation
% START_TIME    : Start time for calculating correlation of CSI.
% DURATION      : Duration of data used for correlation.
%
% CORR_VAL      : Average correlation value per subcarrier, antenna
%

% Calculate correlation coefficient.
corr_val = zeros(n_subcarriers, n_antennas);
corr_space = 5;
sidx = floor(start_time * sample_rate) + 1;
eidx = floor((start_time + duration) * sample_rate);

for jj = 1:n_antennas
    for kk = 1:n_subcarriers
        corr_sum = 0;
        corr_cnt = min(n_subcarriers, kk+corr_space) - max(1,kk-corr_space);
        magic_number = 10;
        corr_map = ones(n_subcarriers, n_subcarriers) * magic_number;
        for ll = max(1,kk-corr_space):min(n_subcarriers, kk+corr_space)
            if ll == kk
                continue;
            end
            if corr_map(ll, kk) == magic_number
                corr_curr = corrcoef(csi_data(sidx:eidx, (jj-1)*n_subcarriers+ll), csi_data(sidx:eidx, (jj-1)*n_subcarriers+kk));
                corr_curr = abs(corr_curr(1,2));
                corr_map(ll, kk) = corr_curr;
                corr_map(kk, ll) = corr_curr;
            else
                corr_curr = corr_map(ll, kk);
            end
            corr_sum = corr_curr + corr_sum;
        end
        corr_val(kk,jj) = corr_sum / corr_cnt;
    end
    
end
end