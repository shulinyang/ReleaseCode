% MAIN
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;
% Set Parameters for Computing Main path
wave_length = 299792458 / 5.825e9;
sample_rate=1024;
main_path_group_number = 60;
path_area=10;
main_path_number = 5;
attenuation_efficient=0.5;
% Set Parameters for Loading Data
total_user=6;
total_gesture = 6;
total_position = 5;
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

data_root = 'F:\wf_tally/';
% data_dir='CSI/20181130/';
% data_type = {'cbh/','cxj/', 'hkh/', 'lsc/', 'ltt/', 'sm/', 'yjj/', 'yumeng/','zk/'};
data_dir='CSI/envs/';
data_type = {'qiankun/','zhangyi/', 'chujing/', 'guidong/', 'wangxu/', 'zhaoyi/'};

data_file_prefix = 'user';
data_pwd = [data_root,data_dir];
anchor_gesture_index=9;
kernel_type='sobel';

% save_type=['LINK6_PCA5_PATH5_ANCHOR',num2str(anchor_gesture_index),'_',kernel_type,'_0.5/'];
% save_type=['LINK6_PCA5_PATH5_TAllYEXT_',kernel_type,'_0.5/'];
save_type='TEST111/';

features_dir = [data_root,'FEATURES/',save_type];
if ~exist(features_dir)
    mkdir(features_dir);
end
tic
for user_index=2:2
    data_path=[data_pwd,data_type{user_index}];
    for gesture_index = 2:total_gesture
        for position_index = 1:total_position
            for orientation_index = 1:total_orientation
                index_for_instance=1;
                for instance_index = 1:total_instance
                    data_file_name = [data_file_prefix,num2str(user_index+8), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index)];
                    
                    for instance_follower_index = instance_index:total_instance
                        data_file_name_follower = [data_file_prefix,num2str(user_index+8), '-', num2str(gesture_index), '-', num2str(position_index),...
                            '-', num2str(orientation_index), '-', num2str(instance_follower_index)];
                        try
                            % Generate Doppler Spectrum
%                             tic
                            [doppler_spectrum, freq_axis_bin,velocity_axis_bin,~,~] = generate_doppler_spectrum_gesture_instance_tailed([data_path, data_file_name],[data_path, data_file_name_follower],...
                                n_receivers, n_antennas, n_subcarriers,wave_length,sample_rate,n_pca);
%                             disp('generate_doppler_spectrum ')
%                             toc
                            % Cyclic Doppler Spectrum According To frequency bin
                            [~,idx] = max(freq_axis_bin);
                            circle_length = length(freq_axis_bin) - idx;
                            doppler_spectrum = circshift(doppler_spectrum, [0 0 circle_length 0]);
                            freq_axis_bin=circshift(freq_axis_bin, [0 circle_length]);
                            velocity_axis_bin=circshift(velocity_axis_bin, [0 circle_length]);
                            
                            %                         doppler_spectrum=cat(4,doppler_spectrum,doppler_spectrum);
                            
                            path_group_size=floor(size(doppler_spectrum,4)/main_path_group_number);
                            if(path_group_size<10)
                                disp([data_file_name,'is too small:',path_group_size])
                                continue;
                            end
                            freq_bin_number=size(doppler_spectrum,3);
                            main_path_freq = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                            velocity_bins_freq=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                            main_path = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                            for kk = 1:n_receivers
                                for mm = 1:n_pca
                                    %
                                    [main_path_freq(kk,mm,:,:),velocity_bins_freq(kk,mm,:,:)] = seam_carving_freq_tailed(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,main_path_number,path_area,kernel_type,'h',velocity_bin,velocity_resolution,attenuation_efficient);
                                    
                                    
                                    %                                     main_path(kk,mm,:,:)=main_path_freq(kk,mm,:,:);
                                    %                                     %                                     disp(['seam_carving_ ',int2str(kk),'_',int2str(mm)])
                                    %                                     %                                     toc
                                    %                                     figure;
                                    %                                     colormap(jet);
                                    %                                     mesh(1:size(doppler_spectrum,4),-60:60,squeeze(doppler_spectrum(kk,mm,:,:)));view([0,90]);
                                    %                                     xlim([0,size(doppler_spectrum,4)]);ylim([-60,60]);
                                    %                                     set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
                                    %                                     set(gcf,'WindowStyle','normal','Position', [0,0,300,250]); % window size
                                    %                                     set (gca,'color','none', 'fontsize', 12); % fontsize
                                    %                                     set(gca,'yTick',-60:20:60);
                                    %                                     xlabel('Time (ms)'); % x label
                                    %                                     ylabel('Frequency (Hz)'); % y label
                                    %                                     colorbar; %Use colorbar only if necessary
                                    %                                     caxis([min(doppler_spectrum(:)),max(doppler_spectrum(:))]);
                                    %                                     saveas(gcf,[data_file_name, '-r', num2str(kk),'-p', num2str(mm),'-velocity', '.png']);
                                    %
                                    %                                     figure;
                                    %                                     colormap(jet);
                                    %                                     mesh(1:size(main_path,4),squeeze(velocity_axis_bin),squeeze(main_path(kk,mm,:,:)));view([0,90]);
                                    %                                     xlim([0,size(main_path,4)]);ylim([min(velocity_axis_bin),max(velocity_axis_bin)]);
                                    %                                     %                                 set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
                                    %                                     set(gcf,'WindowStyle','normal','Position', [0,0,300,250]); % window size
                                    %                                     set (gca,'color','none', 'fontsize', 12); % fontsize
                                    %                                     set(gca,'yTick',-3:0.5:3);
                                    %                                     xlabel('Time (ms)'); % x label
                                    %                                     ylabel('Velocity (m/s)'); % y label
                                    %                                     colorbar; %Use colorbar only if necessary
                                    %                                     %                                 caxis([min(main_path(:)),max(main_path(:))]);
                                    %                                     saveas(gcf,[data_file_name, '-r', num2str(kk),'-p', num2str(mm),'-seam', '.png']);
                                    
                                end
                            end
                            %                             Save VS
%                             toc
                            save_data_file_name = [num2str(user_index+8), '-', num2str(gesture_index), '-', num2str(position_index),...
                                '-', num2str(orientation_index), '-', num2str(index_for_instance), '-'];
                            disp(['Loading ', data_file_name,',',data_file_name_follower,',',save_data_file_name])
                            save([features_dir,save_data_file_name, '.mat'],'velocity_bins_freq');
                            index_for_instance=index_for_instance+1;
                            
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
                end
            end
        end
    end
end
disp([save_type,'is finished'])
toc