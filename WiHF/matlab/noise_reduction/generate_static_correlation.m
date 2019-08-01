function static_corr_val = generate_static_correlation(csi_data,static_exp_path,start_time,end_time,duration, n_antennas, n_subcarriers, sample_rate)
% static_corr_val = generate_static_correlation(csi_data,static_exp_path,start_time,end_time,duration, rx_acnt, n_subcarriers, sample_rate); 
% Return the static correlation matrix, eg:dim(n_subcarriers, n_antennas).
% 
% csi_data: raw CSI measurements, eg:dim(90*1479)
% static_exp_path: the path to save
% start_time: chunk start time
% end_time: chunk end time
% duration: chunk duration
% n_antenna: Antenna count for each receiver, eg:3
% n_subcarriers: Number of suybcarrier, eg:30
% sample_rate: sampling rate
% 
% static_corr_val : the static correlation value.


for ii = start_time:duration:end_time
    if ii == start_time
        static_corr_val = static_correlation(csi_data, n_antennas, n_subcarriers, sample_rate, start_time, duration);
    else
        static_corr_val = static_corr_val + static_correlation(csi_data, n_antennas, n_subcarriers, sample_rate, ii, duration);
    end
end
start_time_list = start_time:duration:end_time;
static_corr_val = static_corr_val / length(start_time_list);
save(static_exp_path, 'static_corr_val');
end