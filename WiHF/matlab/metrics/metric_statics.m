% VALIDATE_HYBRID)PATH_TYPE
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;

% Set Parameters for Computing Main path
wave_length = 299792458 / 5.825e9;
sample_rate=1024;
main_path_group_number = 30;
path_area=10;
main_path_number = 5;
attenuation_efficient=0.5;
% Set Parameters for Loading Data
total_user=6;
total_gesture = 8;
total_position = 1;
total_orientation = 5;
total_instance = 5;
n_receivers = 6;     % Receiver count(no less than 3)
n_antennas = 3;    % Antenna count for each receiver
n_subcarriers=30;
n_pca=3;

%Set Parameters for Velocity Mapping
velocity_upper_bound = 1.6;
velocity_lower_bound = -1.6;
velocity_number = 20;
velocity_resolution = (velocity_upper_bound - velocity_lower_bound)/velocity_number;
velocity_bin = ((1:velocity_number) - velocity_number/2) / (velocity_number/2) * velocity_upper_bound;

% Configuration for Dataset
data_root = 'F:\wf_tally/';
data_dir='CSI/20181130/';
data_type = {'cbh/','cxj/', 'hkh/', 'lsc/', 'ltt/', 'sm/', 'yjj/', 'yumeng/','zk/'};

% data_dir='CSI/envs/';
% data_type = {'qiankun/','zhangyi/', 'chujing/', 'guidong/', 'wangxu/', 'zhaoyi/'};
data_file_prefix = 'user';
kernel_type='sobel';
data_pwd = [data_root,data_dir];
save_type='TALLY_BVP/';

metrics_dir = [data_root,'METRICS/',save_type];
if ~exist(metrics_dir)
    mkdir(metrics_dir);
end


sample_length_time = zeros(total_user*total_gesture*total_position*total_orientation*total_instance,4);
time_index=0;
for user_index=1:total_user
    data_path=[data_pwd,data_type{user_index}];
    for gesture_index = 1:total_gesture
        for position_index = 1:total_position
            for orientation_index = 1:total_orientation
                for instance_index = 1:total_instance
                    data_file_name = [data_file_prefix,num2str(user_index), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index)];
                    disp(['Loading ', data_file_name])
                    try
                        time_index=time_index+1;
                        % Generate Doppler Spectrum
                        [doppler_spectrum, freq_axis_bin,velocity_axis_bin,sample_length_time(time_index,1),sample_length_time(time_index,2)] = generate_doppler_spectrum([data_path, data_file_name],...
                            n_receivers, n_antennas, n_subcarriers,'stft',wave_length,sample_rate,n_pca);
                        sample_length_time(time_index,4)=size(doppler_spectrum,4)/1000;
                        tic;
                        % Cyclic Doppler Spectrum According To frequency bin
                        [~,idx] = max(freq_axis_bin);
                        circle_length = length(freq_axis_bin) - idx;
                        doppler_spectrum = circshift(doppler_spectrum, [0 0 circle_length 0]);
                        freq_axis_bin=circshift(freq_axis_bin, [0 circle_length]);
                        velocity_axis_bin=circshift(velocity_axis_bin, [0 circle_length]);
           
                        path_group_size=floor(size(doppler_spectrum,4)/main_path_group_number);
                        if(path_group_size<5)
                            disp([data_file_name,'is too small:',path_group_size])
                            continue;
                        end
                        freq_bin_number=size(doppler_spectrum,3);
                        main_path_freq = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        velocity_bins_freq=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        for kk = 1:n_receivers
                            for mm = 1:n_pca
                                [main_path_freq(kk,mm,:,:),velocity_bins_freq(kk,mm,:,:)] = seam_carving_freq_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'h',velocity_bin,velocity_resolution,attenuation_efficient);
                            end
                        end
                        sample_length_time(time_index,3)=toc/n_receivers;                    
                    catch err
                        exception_fid = fopen([metrics_dir, 'exception_log_',data_file_name,'.log'],'wt');
                        fprintf(exception_fid, '%s\n', date);
                        disp(['Exception Occured: ' err.message]);
                        fprintf(exception_fid, '%s\n', data_file_name);
                        fprintf(exception_fid, '%s\n', err.message);
                        fclose(exception_fid);
                        continue;
                    end
                end
            end
        end
    end
end
save([metrics_dir, 'time_consuming.mat'], 'sample_length_time');
toc