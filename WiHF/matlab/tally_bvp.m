% MAIN
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
total_user=8;
total_gesture = 6;
total_position = 1;
total_orientation = 1;
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

data_root = 'F:\wf_tally/';
data_dir='CSI/20181130/';
data_type = {'cbh/','cxj/', 'hkh/', 'lsc/', 'ltt/', 'sm/', 'yjj/', 'yumeng/','zk/'};

% data_dir='CSI/envs/';
% data_type = {'qiankun/','zhangyi/', 'chujing/', 'guidong/', 'wangxu/', 'zhaoyi/'};

data_file_prefix = 'user';
kernel_type='sobel';
data_pwd = [data_root,data_dir];
% save_type=['LINK6_PCA5_PATH5_BVP_TAILED_',kernel_type,'_0.5/'];
save_type='TEST/';

features_dir = [data_root,'FEATURES/',save_type];
if ~exist(features_dir)
    mkdir(features_dir);
end

% metrics_dir = [data_root,'METRICS/',save_type];
% if ~exist(metrics_dir)
%     mkdir(metrics_dir);
% end

% set(0,'DefaultFigureVisible', 'off')
tic

% velo_sequence_freq_dist = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);
% velo_sequence_time_dist = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);
% velo_sequence_power_dist = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);

% velo_sequence_freq_data = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,n_pca, 121);
% velo_sequence_time_data = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,n_pca, main_path_group_number);
% velo_sequence_power_data = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,n_pca,main_path_group_number);
user_fig_index=0;
for user_index=1:total_user
    data_path=[data_pwd,data_type{user_index}];
    for gesture_index = 1:total_gesture
        for position_index = 1:total_position
            for orientation_index = 1:total_orientation
                for instance_index = 1:total_instance
                    data_file_name = [data_file_prefix,num2str(user_index+8), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index)];
                    disp(['Loading ', data_file_name])
                    try
                        % Generate Doppler Spectrum
                        %                         tic
                        [doppler_spectrum, freq_axis_bin,velocity_axis_bin,~,~] = generate_doppler_spectrum([data_path, data_file_name],...
                            n_receivers, n_antennas, n_subcarriers,'stft',wave_length,sample_rate,n_pca);
                        %                         disp('generate_doppler_spectrum ')
                        %                         toc
                        % Cyclic Doppler Spectrum According To frequency bin
                        [~,idx] = max(freq_axis_bin);
                        circle_length = length(freq_axis_bin) - idx;
                        doppler_spectrum = circshift(doppler_spectrum, [0 0 circle_length 0]);
                        freq_axis_bin=circshift(freq_axis_bin, [0 circle_length]);
                        velocity_axis_bin=circshift(velocity_axis_bin, [0 circle_length]);
                        
                        %                         doppler_spectrum=cat(4,doppler_spectrum,doppler_spectrum);
                        
                        %                         path_group_size=floor(size(doppler_spectrum,4)/main_path_group_number);
                        path_group_size=floor(size(doppler_spectrum,4)/main_path_group_number);
                        if(path_group_size<5)
                            disp([data_file_name,'is too small:',path_group_size])
                            continue;
                        end
                        freq_bin_number=size(doppler_spectrum,3);
                        %                         main_path_sum = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        %                         velocity_bins_sum=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        %
                        %                         main_path_time = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        %                         velocity_bins_time=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        %
                        %                         main_path_time_reverse = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        %                         velocity_bins_time_reverse=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path_freq = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        velocity_bins_freq=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        %                         main_path_freq_reverse = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        %                         velocity_bins_freq_reverse=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        %
                        %                         main_path_power = zeros(n_receivers,n_pca,freq_bin_number, main_path_group_number);
                        %                         velocity_bins_power=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        
                        
                        
                        for kk = 1:n_receivers
                            for mm = 1:n_pca
                                %                                 [main_path_sum(kk,mm,:,:),velocity_bins_sum(kk,mm,:,:)] = seam_carving_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'v',velocity_bin,velocity_resolution,attenuation_efficient);
                                %
                                %                                 [main_path_time(kk,mm,:,:),velocity_bins_time(kk,mm,:,:)] = seam_carving_time_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'v',velocity_bin,velocity_resolution,attenuation_efficient);
                                %                                 [main_path_time_reverse(kk,mm,:,:),velocity_bins_time_reverse(kk,mm,:,:)] = seam_carving_time_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'h',velocity_bin,velocity_resolution,attenuation_efficient);
                                %
                                [main_path_freq(kk,mm,:,:),velocity_bins_freq(kk,mm,:,:)] = seam_carving_freq_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'h',velocity_bin,velocity_resolution,attenuation_efficient);
                                %                                 [main_path_freq_reverse(kk,mm,:,:),velocity_bins_freq_reverse(kk,mm,:,:)] = seam_carving_freq(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'v',velocity_bin,velocity_resolution,attenuation_efficient);
                                %
                                %                                 [main_path_power(kk,mm,:,:),velocity_bins_power(kk,mm,:,:)] = graph_matching_multipath(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,velocity_bin,velocity_resolution,attenuation_efficient);
                                
                                
                                %                                 main_path(kk,mm,:,:)=main_path_freq(kk,mm,:,:)+main_path_time(kk,mm,:,:);
                                %                                 %                                 disp(['seam_carving_ ',int2str(kk),'_',int2str(mm)])
                                %                                 %                                 toc
                                %                                 figure;
                                %                                 disp([data_file_name, '-r', num2str(kk),'-p', num2str(mm)])
                                %                                 colormap(jet);
                                %                                 mesh(1:size(doppler_spectrum,4),-60:60,squeeze(doppler_spectrum(kk,mm,:,:)));view([0,90]);
                                %                                 xlim([0,size(doppler_spectrum,4)]);ylim([-60,60]);
                                %                                 %                                 set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
                                %                                 set(gcf,'WindowStyle','normal','Position', [300*(instance_fig_index-1),300*(user_fig_index-1),300,250]); % window size
                                %                                 set (gca,'color','none', 'fontsize', 12); % fontsize
                                %                                 set(gca,'yTick',-60:20:60);
                                %                                 xlabel('Time (ms)'); % x label
                                %                                 ylabel('Frequency (Hz)'); % y label
                                %                                 colorbar; %Use colorbar only if necessary
                                %                                 caxis([min(doppler_spectrum(:)),max(doppler_spectrum(:))]);
                                %                                 %                                 saveas(gcf,[data_file_name, '-r', num2str(kk),'-p', num2str(mm),'-velocity', '.png']);
                                %
                                %                                 figure;
                                %                                 colormap(jet);
                                %                                 mesh(1:size(main_path,4),squeeze(velocity_axis_bin),squeeze(main_path(kk,mm,:,:)));view([0,90]);
                                %                                 xlim([0,size(main_path,4)]);ylim([min(velocity_axis_bin),max(velocity_axis_bin)]);
                                %                                 %                                 set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
                                %                                 set(gcf,'WindowStyle','normal','Position', [0,0,300,250]); % window size
                                %                                 set (gca,'color','none', 'fontsize', 12); % fontsize
                                %                                 set(gca,'yTick',-3:0.5:3);
                                %                                 xlabel('Time (ms)'); % x label
                                %                                 ylabel('Velocity (m/s)'); % y label
                                %                                 colorbar; %Use colorbar only if necessary
                                %                                 caxis([min(main_path(:)),max(main_path(:))]);
                                %                                 saveas(gcf,[data_file_name, '-r', num2str(kk),'-p', num2str(mm),'-seam', '.png']);
                                %
                                %
                                %                                 %                                 velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,:,:,:,:)=velo_sequence_freq;
                                %                                 %                                 velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,:,:,:,:)=velo_sequence_time;
                                %                                 %                                 velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,:,:,:,:)=velo_sequence_power;
                            end
                        end
                        %                         velocity_bins_freq_sum=velocity_bins_time+velocity_bins_freq;
                        %                         velocity_bins_freq_sum_reverse=velocity_bins_time_reverse+velocity_bins_freq_reverse;
                        %                         Save VS
                        %                                                 toc
                        %                         save([features_dir,data_file_name, '.mat'],'velocity_bins_sum', 'velocity_bins_time','velocity_bins_freq','velocity_bins_freq_sum','velocity_bins_freq_sum_reverse','velocity_bins_time_reverse','velocity_bins_freq_reverse','velocity_bins_power');
                        save([features_dir,num2str(user_index+8), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index), '-.mat'],'velocity_bins_freq');
                        
                    catch err
                        exception_fid = fopen([features_dir, 'exception_log_',data_file_name,'.log'],'wt');
                        fprintf(exception_fid, '%s\n', date);
                        disp(['Exception Occured: ' err.message]);
                        fprintf(exception_fid, '%s\n', data_file_name);
                        fprintf(exception_fid, '%s\n', err.message);
                        fclose(exception_fid);
                        continue;
                    end
                end
                %                 for kk = 1:n_receivers
                %                     for mm = 1:pca_number
                %                         velo_sequence_freq_dist(user_sel,gesture_sel,position_sel,orientation_sel,kk,mm,:) = pdist(squeeze(velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,:,kk,mm,:)),'minkowski')';
                %                         velo_sequence_time_dist(user_sel,gesture_sel,position_sel,orientation_sel,kk,mm,:)  =pdist(squeeze(velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,:,kk,mm,:)),'minkowski')';
                %                         velo_sequence_power_dist(user_sel,gesture_sel,position_sel,orientation_sel,kk,mm,:)  = pdist(squeeze(velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,:,kk,mm,:)),'minkowski')';
                %                     end
                %                 end
            end
        end
    end
end
% save([metrics_dir,data_file_name, '.mat'], 'velo_sequence_freq_data','velo_sequence_time_data','velo_sequence_power_data');
disp([save_type,'is finished'])
toc