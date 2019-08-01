function [main_path,velocity_mapping_feature] = seam_carving_tailed_path(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,kernel_type,path_type,velocity_bin,velocity_resolution,attenuation_efficient)
% [main_path,velocity_mapping_feature,velocity_sequence] = seam_carving_time(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,kernel_type,path_type,velocity_bin,velocity_resolution,attenuation_efficient)
% extracts velocity feature and sequence along time axis from spectrogram using kernel
%
% FTP         : The spectrogram.
% ftp_list   : tha path value candidates.
% path_sample_rate   : Sampling rate of PLCR series.
% SAMPLE_RATE : Sampling rate of spectrogram.
%
% path_matched        : Path matched.
% TFA With CWT or STFT
switch kernel_type
    case 'scharr'
        kernel_h =[ 3, 0, -3;
            10, 0, -10;
            3, 0, -3];
        
        kernel_v =[3, 10, 3;
            0, 0, 0;
            -3,-10,-3];
    case 'sobel'
        kernel_h =[ 1, 0, -1;
            2, 0, -2;
            1, 0, -1];
        kernel_v =[1, 2, 1;
            0, 0, 0;
            -1,-2,-1];
    case {'laplace'}
        kernel_h =[ 0, 1, 0;
            1, -4, 1;
            0, 1, 0];
        kernel_v =[0, 1, 0;
            1, -4, 1;
            0, 1, 0];
    case {'robert'}
        kernel_h =[ 0, 0, 0;
            0, 1, -1;
            0, 0, 0];
        kernel_v =[0, 0, 0;
            0, 1, 0;
            0, -1, 0];
    otherwise
        warning('Unexpected kernal.')
end

doppler_spectrum_grouped = zeros(size(doppler_spectrum,1), time_axis_number);
main_path=zeros(size(doppler_spectrum_grouped,1),time_axis_number);
for trace_index = 1:size(doppler_spectrum_grouped,2)
    doppler_spectrum_grouped(:,trace_index) = mean(doppler_spectrum(:,(trace_index-1)*path_group_size+1:trace_index*path_group_size),2);
end

freq_index = zeros(main_path_number, time_axis_number);
% velocity_sequence = zeros(main_path_number, time_axis_number);
velocity_mapping_feature = zeros(size(velocity_bin,2), time_axis_number);
power_matrix=doppler_spectrum_grouped;
for path_index=1:main_path_number
    gradiant_h = imfilter(power_matrix,kernel_h,'conv');
    gradiant_v = imfilter(power_matrix,kernel_v,'conv');
    gradiant_matrix=abs(gradiant_h)+abs(gradiant_v);
    
    power_matrix_sum = zeros(size(gradiant_matrix));
    power_trace = zeros(size(gradiant_matrix));
    for trace_index = 1:size(gradiant_matrix, 2)
        if trace_index == 1
            power_matrix_sum(:,trace_index) = gradiant_matrix(:,trace_index);
            power_trace(:,trace_index) = 0;
        else
            for jj = 1:size(gradiant_matrix, 1)
                if jj == 1
                    [last_pow, idx] = max(power_matrix_sum(jj:jj+1,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-1;
                elseif jj == size(gradiant_matrix, 1)
                    [last_pow, idx] = max(power_matrix_sum(jj-1:jj,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-2;
                else
                    [last_pow, idx] = max(power_matrix_sum(jj-1:jj+1,trace_index-1));
                    power_trace(jj,trace_index) = jj+idx-2;
                end
                power_matrix_sum(jj,trace_index) = last_pow + gradiant_matrix(jj,trace_index);
            end
        end
    end
    
    [~, freq_index(path_index,end)] = max(power_matrix_sum(:,end));
    power_matrix(max(freq_index(path_index,end)-path_area/2,1):min(freq_index(path_index,end)+path_area/2,size(power_matrix_sum,1)),end)=power_matrix(max(freq_index(path_index,end)-path_area/2,1):min(freq_index(path_index,end)+path_area/2,size(power_matrix_sum,1)),end)*attenuation_efficient;
    trace_index = size(power_matrix_sum,2);
    while trace_index > 1
        freq_index(path_index,trace_index-1) = power_trace(freq_index(path_index,trace_index),trace_index);
        power_matrix(max(freq_index(path_index,trace_index-1)-path_area/2,1):min(freq_index(path_index,trace_index-1)+path_area/2,size(power_matrix_sum,1)),trace_index)=power_matrix(max(freq_index(path_index,trace_index-1)-path_area/2,1):min(freq_index(path_index,trace_index-1)+path_area/2,size(power_matrix_sum,1)),trace_index)*attenuation_efficient;
        main_path(max(freq_index(path_index,trace_index-1)-1,1):min(freq_index(path_index,trace_index-1)+1,size(power_matrix_sum,1)),trace_index)=1;
        trace_index = trace_index - 1;
    end
    
    for time_axis_index=1:time_axis_number
        velocity_axis_index=freq_index(path_index,time_axis_index);
        %         velocity_sequence(path_index,time_axis_index)=doppler_spectrum_grouped(velocity_axis_index,time_axis_index);
        velo_trun=velocity_axis_bin(1,velocity_axis_index);
        velocity_axis_index_trun=ceil(velo_trun/velocity_resolution+size(velocity_bin,2)/2);
        %         if(velocity_axis_index_trun<1 || velocity_axis_index_trun>size(velocity_bin,2))
        %             continue;
        %         end
        main_path(velocity_axis_index,time_axis_index)=1;
        velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)=velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)+main_path(velocity_axis_index,time_axis_index);
        %         main_path(velocity_axis_index,time_axis_index)=main_path(velocity_axis_index,time_axis_index)+gradiant_matrix(velocity_axis_index,time_axis_index);
        %         velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)=velocity_mapping_feature(velocity_axis_index_trun,time_axis_index)+main_path(velocity_axis_index,time_axis_index);
    end
end
end