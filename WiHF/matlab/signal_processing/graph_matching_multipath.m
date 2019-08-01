function [main_path,velocity_mapping_feature,velocity_sequence] = graph_matching_multipath(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,velocity_bin,V_resolution,attenuation_efficient)
% [main_path,velocity_mapping_feature,velocity_sequence] = graph_matching_multipath(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,velocity_bin,V_resolution,attenuation_efficient)
% extracts velocity feature and sequence along time axis from spectrogram
% using power
%
% FTP         : The spectrogram.
% ftp_list   : tha path value candidates.
% path_sample_rate   : Sampling rate of PLCR series.
% SAMPLE_RATE : Sampling rate of spectrogram.
%
% path_matched        : Path matched.
% TFA With CWT or STFT
freq_axis_number=size(doppler_spectrum,1);
doppler_spectrum_grouped = zeros(freq_axis_number, time_axis_number);
main_path=zeros(size(doppler_spectrum_grouped));
for trace_index = 1:time_axis_number
    doppler_spectrum_grouped(:,trace_index) = mean(doppler_spectrum(:,(trace_index-1)*path_group_size+1:trace_index*path_group_size),2);
end

freq_index = zeros(main_path_number, time_axis_number);
velocity_sequence = zeros(main_path_number, time_axis_number);
% velo_sequence_bins = zeros(main_path_number, time_axis_number);
velocity_mapping_feature = zeros(size(velocity_bin,2), size(doppler_spectrum_grouped,2));
power_matrix=doppler_spectrum_grouped;

for path_index=1:main_path_number    
    power_matrix_sum = zeros(size(power_matrix));
    power_trace = zeros(size(power_matrix));
    for trace_index = 1:size(power_matrix, 2)
        if trace_index == 1
            power_matrix_sum(:,trace_index) = power_matrix(:,trace_index);
            power_trace(:,trace_index) = 0;
        else
            for jj = 1:size(power_matrix, 1)
                if jj == 1
                    [last_pow, idx] = max(power_matrix_sum(jj:jj+1,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-1;
                elseif jj == size(power_matrix, 1)
                    [last_pow, idx] = max(power_matrix_sum(jj-1:jj,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-2;
                else
                    [last_pow, idx] = max(power_matrix_sum(jj-1:jj+1,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-2;
                end
                power_matrix_sum(jj,trace_index) = last_pow + power_matrix(jj,trace_index);
            end
        end
    end
    
    [~, freq_index(path_index,end)] = max(power_matrix_sum(:,end));
    for path_area_index=max(freq_index(path_index,end)-path_area/2,1):min(freq_index(path_index,end)+path_area/2,size(power_matrix_sum,1))
        power_matrix(path_area_index,end)=power_matrix(path_area_index,end)*attenuation_efficient;
    end
    trace_index = size(power_matrix_sum,2);
    while trace_index > 1
        freq_index(path_index,trace_index-1) = power_trace(freq_index(path_index,trace_index),trace_index);
        for path_area_index=max(freq_index(path_index,trace_index-1)-path_area/2,1):min(freq_index(path_index,trace_index-1)+path_area/2,size(power_matrix_sum,1))
            power_matrix(path_area_index,trace_index)=power_matrix(path_area_index,trace_index)*attenuation_efficient;
        end
        trace_index = trace_index - 1;
    end
    
    for time_axis_index=1:time_axis_number
        index=freq_index(path_index,time_axis_index);
%         velo_sequence_bins(path_index,time_axis_index)=index/ size(doppler_spectrum_grouped,1);
        velocity_sequence(path_index,time_axis_index)=doppler_spectrum_grouped(index,time_axis_index);
        velo_trun=velocity_axis_bin(1,index);
        velocity_axis_index_trun=floor(velo_trun/V_resolution+size(velocity_bin,2)/2);
        if(velocity_axis_index_trun<1)
            velocity_axis_index_trun=1;
        end
        if(velocity_axis_index_trun>size(velocity_bin,2))
            velocity_axis_index_trun=size(velocity_bin,2);
        end
        main_path(index,time_axis_index)=main_path(index,time_axis_index)+doppler_spectrum_grouped(index,time_axis_index);
        velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)=velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)+main_path(index,time_axis_index);
    end
end
end