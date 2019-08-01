% GENERATE_DUPLICATE_CSI
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;

% Set Parameters for Loading Data
wave_length = 299792458 / 5.825e9;
sample_rate=1024;
total_user=9;
total_gesture = 9;
total_position = 5;
total_orientation = 5;
total_instance = 5;
n_receivers = 6;
n_antennas = 3;
n_subcarriers=30;

data_root = 'F:\wf_tally/';
data_dir='CSI/20181130/';
data_file_prefix = 'user';
data_type = {'cbh/','cxj/', 'hkh/', 'lsc/', 'ltt/', 'sm/', 'yjj/', 'yumeng/','zk/'};
data_pwd = [data_root,data_dir];
save_type='LINK6_ANCHOR_CSI/';
% save_type='TEST/';
hybrid_dir = [data_root,'HYBRID/',save_type];
if ~exist(hybrid_dir)
    mkdir(hybrid_dir);
end

tic
for user_index=1:total_user
    data_path=[data_pwd,data_type{user_index}];
    for gesture_index = 1:total_gesture
        for position_index = 1:total_position
            for orientation_index = 1:total_orientation
                index=1;
                for instance_index = 1:total_instance-1
                    data_file_name_header = [data_file_prefix,num2str(user_index), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index)];
                    for ii = 1:n_receivers
                        csi_path = [data_path, data_file_name_header, '-r', num2str(ii), '.dat'];
                        try
                            [csi_data,~] =  generate_csi_from_dat(csi_path,n_antennas,n_subcarriers);
                            if(ii==1)
                                csi_data_header = zeros(n_receivers,size(csi_data, 1),size(csi_data, 2));
                            end
                            if(size(csi_data,1) >= size(csi_data_header,2))
                                csi_data_header(ii,:,:) = csi_data(1:size(csi_data_header,2),:);
                            else
                                csi_data_header(ii,:,:) = cat(1,csi_data,zeros(size(csi_data_header,2) - size(csi_data,1),size(csi_data_header,3)));
                            end
                        catch err
                            disp(err)
                            continue
                        end
                    end
                    
                    for instance_follower_index = instance_index+1:total_instance
                        data_file_name_follower = [data_file_prefix,num2str(user_index), '-', num2str(gesture_index), '-', num2str(position_index),...
                            '-', num2str(orientation_index), '-', num2str(instance_follower_index)];
                        for ii = 1:n_receivers
                            csi_path = [data_path, data_file_name_follower, '-r', num2str(ii), '.dat'];
                            try
                                [csi_data,~] =  generate_csi_from_dat(csi_path,n_antennas,n_subcarriers);
                                if(ii==1)
                                    csi_data_follower = zeros(n_receivers,size(csi_data, 1),size(csi_data, 2));
                                end
                                if(size(csi_data,1) >= size(csi_data_follower,2))
                                    csi_data_follower(ii,:,:) = csi_data(1:size(csi_data_follower,2),:);
                                else
                                    csi_data_follower(ii,:,:) = cat(1,csi_data,zeros(size(csi_data_follower,2) - size(csi_data,1),size(csi_data_follower,3)));
                                end
                            catch err
                                disp(err)
                                continue
                            end
                        end
                        disp([data_file_name_header,data_file_name_follower])
                        csi_data=cat(2,csi_data_header,csi_data_follower);
                        sdata_file_name = [data_file_prefix,num2str(user_index), '-', num2str(gesture_index), '-', num2str(position_index),...
                            '-', num2str(orientation_index), '-', num2str(index)];
                        save([hybrid_dir,sdata_file_name, '.mat'], 'csi_data');%(n_receiver,time_sequence,n_antennas*n_subcarrier)
                        index=index+1;
                    end
                end
            end
        end
    end
end
toc