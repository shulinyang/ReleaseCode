function [main_path,velocity_mapping_feature] = seam_carving_freq_tailed(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,kernel_type,path_type,velocity_bin,V_resolution,attenuation_efficient)
% [main_path,velocity_mapping_feature,velocity_sequence] = seam_carving_freq(doppler_spectrum, velocity_axis_bin, time_axis_number,path_group_size,main_path_number,path_area,kernel_type,path_type,velocity_bin,V_resolution,attenuation_efficient)
% extracts velocity feature and sequence along freq axis from spectrogram using kernel
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
        if path_type=='h'
            kernel =[ 3, 0, -3;
                10, 0, -10;
                3, 0, -3];
        else
            kernel =[3, 10, 3;
                0, 0, 0;
                -3,-10,-3];
        end
    case 'sobel'
        if path_type=='h'
            kernel =[ 1, 0, -1;
                2, 0, -2;
                1, 0, -1];
        else
            kernel =[1, 2, 1;
                0, 0, 0;
                -1,-2,-1];
        end
    case {'laplace'}
        if path_type=='h'
            kernel =[ 0, 1, 0;
                1, -4, 1;
                0, 1, 0];
        else
            kernel =[0, 1, 0;
                1, -4, 1;
                0, 1, 0];
        end
    case {'robert'}
        if path_type=='h'
            kernel =[ 0, 0, 0;
                0, 1, -1;
                0, 0, 0];
        else
            kernel =[0, 0, 0;
                0, 1, 0;
                0, -1, 0];
        end
    otherwise
        warning('Unexpected kernal.')
end

freq_axis_number=size(doppler_spectrum,1);
doppler_spectrum_grouped = zeros(freq_axis_number, time_axis_number);
for trace_index = 1:time_axis_number
    doppler_spectrum_grouped(:,trace_index) = mean(doppler_spectrum(:,(trace_index-1)*path_group_size+1:trace_index*path_group_size),2);
end
doppler_spectrum_grouped=doppler_spectrum_grouped';
main_path=zeros(size(doppler_spectrum_grouped));
freq_index = zeros(main_path_number, freq_axis_number);
velocity_mapping_feature = zeros(size(velocity_bin,2), time_axis_number);
power_matrix=doppler_spectrum_grouped;
for path_index=1:main_path_number
    gradiant = imfilter(power_matrix,kernel,'conv');
    gradiant_matrix=abs(gradiant);
    
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
        
        trace_index = trace_index - 1;
    end
    
    for freq_axis_index=1:freq_axis_number
        index=freq_index(path_index,freq_axis_index);
        velo_trun=velocity_axis_bin(1,freq_axis_index);
        velocity_axis_index_trun=ceil(velo_trun/V_resolution+size(velocity_bin,2)/2);
        %         if(velocity_axis_index_trun<1 || velocity_axis_index_trun>size(velocity_bin,2))
        %             continue;
        %         end
        main_path(index,freq_axis_index)=main_path(index,freq_axis_index)+doppler_spectrum_grouped(index,freq_axis_index);
        velocity_mapping_feature(velocity_axis_index_trun,index)=velocity_mapping_feature(velocity_axis_index_trun,index)+main_path(index,freq_axis_index);
        %         main_path(index,freq_axis_index)=main_path(index,freq_axis_index)+gradiant_matrix(index,freq_axis_index);
        %         velocity_mapping_feature(velocity_axis_index_trun,index)=velocity_mapping_feature(velocity_axis_index_trun,index)+main_path(index,freq_axis_index);
    end
end
main_path=main_path';
end