clear;
clc;
close all;

dpth_pwd = 'F:\widar_3/';
save_name='LINK6_PCA5_PATH1_DIST_0.5';
dist_dir = [dpth_pwd,'dist/',save_name];

velo_sequence_freq_data=cell2mat(struct2cell(load([dist_dir, '.mat'], 'velo_sequence_freq_data')));
velo_sequence_time_data=cell2mat(struct2cell(load([dist_dir, '.mat'], 'velo_sequence_time_data')));
velo_sequence_power_data=cell2mat(struct2cell(load([dist_dir, '.mat'], 'velo_sequence_power_data')));

total_user=9;% Total user count,max=9
total_gesture = 9;   % Total gesture count,max=9
total_position = 5;  % Total position count,max=5
total_orientation = 5;  % Total orientation count,max=5
total_instance = 5;  % Total instance (gesture repeatation) count,max=5

n_receivers = 6;     % Receiver count(no less than 3)
main_path_number = 1;    % Antenna count for each receiver
pca_number=5;

% velo_sequence_freq_dist = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,10);
% velo_sequence_time_dist = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,10);
% velo_sequence_power_dist = zeros(total_user,total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,10);


velo_sequence_freq_dist_instance = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);
velo_sequence_time_dist_instance = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);
velo_sequence_power_dist_instance = zeros(total_user,total_gesture,total_position,total_orientation,n_receivers,pca_number,10);

velo_sequence_freq_dist_user = zeros(total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,36);
velo_sequence_time_dist_user = zeros(total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,36);
velo_sequence_power_dist_user = zeros(total_gesture,total_position,total_orientation,total_instance,n_receivers,pca_number,36);

velo_sequence_freq_dist_gesture = zeros(total_user,total_position,total_orientation,total_instance,n_receivers,pca_number,36);
velo_sequence_time_dist_gesture = zeros(total_user,total_position,total_orientation,total_instance,n_receivers,pca_number,36);
velo_sequence_power_dist_gesture = zeros(total_user,total_position,total_orientation,total_instance,n_receivers,pca_number,36);
for position_sel = 1:total_position
    for orientation_sel = 1:total_orientation
        for link_sel = 1:n_receivers
            for pca_sel = 1:pca_number
%                 for user_sel=1:total_user
%                     for gesture_sel = 1:total_gesture
%                         for instance_sel = 1:total_instance
%                             norm_freq=velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)-min(velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                             velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)=norm_freq./max(velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                             
%                             norm_time=velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)-min(velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                             velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)=norm_time./max(velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                             
%                             norm_power=velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)-min(velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                             velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)=norm_power./max(velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:));
%                         end
%                     end
%                 end
                
                for user_sel=1:total_user
                    for gesture_sel = 1:total_gesture
                        velo_sequence_freq_dist_instance(user_sel,gesture_sel,position_sel,orientation_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_freq_data(user_sel,gesture_sel,position_sel,orientation_sel,:,link_sel,pca_sel,:)),'cosine')';
                        velo_sequence_time_dist_instance(user_sel,gesture_sel,position_sel,orientation_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_time_data(user_sel,gesture_sel,position_sel,orientation_sel,:,link_sel,pca_sel,:)),'cosine');
                        velo_sequence_power_dist_instance(user_sel,gesture_sel,position_sel,orientation_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_power_data(user_sel,gesture_sel,position_sel,orientation_sel,:,link_sel,pca_sel,:)),'cosine')';
                    end
                end
                for gesture_sel = 1:total_gesture
                    for instance_sel = 1:total_instance
                        velo_sequence_freq_dist_user(gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_freq_data(:,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                        velo_sequence_time_dist_user(gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_time_data(:,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                        velo_sequence_power_dist_user(gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_power_data(:,gesture_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                    end
                end
                for user_sel=1:total_user
                    for instance_sel = 1:total_instance
                        velo_sequence_freq_dist_gesture(user_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_freq_data(user_sel,:,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                        velo_sequence_time_dist_gesture(user_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_time_data(user_sel,:,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                        velo_sequence_power_dist_gesture(user_sel,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:) = pdist(squeeze(velo_sequence_power_data(user_sel,:,position_sel,orientation_sel,instance_sel,link_sel,pca_sel,:)),'cosine')';
                    end
                end
            end
        end
    end
end
figure()
cdfplot(reshape(velo_sequence_freq_dist_instance,[],1))
hold on
cdfplot(reshape(velo_sequence_time_dist_instance,[],1))
cdfplot(reshape(velo_sequence_power_dist_instance,[],1))
grid on
title('Empirical CDF of Main Path for Instance')
legend('Freq','Time','Power')
hold off

figure()
cdfplot(reshape(velo_sequence_freq_dist_user,[],1))
hold on
cdfplot(reshape(velo_sequence_time_dist_user,[],1))
cdfplot(reshape(velo_sequence_power_dist_user,[],1))
grid on
title('Empirical CDF of Main Path for User')
legend('Freq','Time','Power')
hold off

figure()
cdfplot(reshape(velo_sequence_freq_dist_gesture,[],1))
hold on
cdfplot(reshape(velo_sequence_time_dist_gesture,[],1))
cdfplot(reshape(velo_sequence_power_dist_gesture,[],1))
grid on
title('Empirical CDF of Main Path for Gesture')
legend('Freq','Time','Power')
hold off
















