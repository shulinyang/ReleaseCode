% MAIN
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;

% Set Parameters for Computing Main path
wave_length = 299792458 / 5.825e9;
sample_rate=1024;
path_area=20;
main_path_number = 1;
attenuation_efficient=0.5;
% Set Parameters for Loading Data
total_user=9;
total_gesture = 6;
total_position = 5;
total_orientation = 5;
total_instance = 5;
n_receivers = 2;     % Receiver count(no less than 3)
n_antennas = 3;    % Antenna count for each receiver
n_subcarriers=30;
n_pca=5;

%Set Parameters for Velocity Mapping
velocity_upper_bound = 1.6;
velocity_lower_bound = -1.6;
velocity_number = 20;
velocity_resolution = (velocity_upper_bound - velocity_lower_bound)/velocity_number;
velocity_bin = ((1:velocity_number) - velocity_number/2) / (velocity_number/2) * velocity_upper_bound;

data_root = 'F:\wf_tally/';
data_dir='CSI/20181130/';
data_file_prefix = 'user';
data_type = {'cbh/','cxj/', 'hkh/', 'lsc/', 'ltt/', 'sm/', 'yjj/', 'yumeng/','zk/'};
kernel_type='sobel';
data_pwd = [data_root,data_dir];
% save_type=['LINK6_PCA5_PATH5_BVP_',kernel_type,'_0.5/'];
save_type='fig2';

figures_dir = [data_root,'FIGURES/',save_type,'/'];
if ~exist(figures_dir)
    mkdir(figures_dir);
end

save_path=[figures_dir,'data.mat'];

user_fig_index=0;
if ~exist(save_path,'file')==0
    plot_line_h=cell2mat(struct2cell(load(save_path, 'plot_line_h')));
    plot_line_v=cell2mat(struct2cell(load(save_path, 'plot_line_v')));
else
    plot_line_h=ones(3,3,3,3,200)*0.1;
    plot_line_v=ones(3,3,3,3,30)*0.1;
end

N = 5;
color_dot = [ 0 0 0;0.6769,0.4447,0.7114;0.9153,0.2816,0.2878;1,0.75,0.79];

label_legend={'Power','Upper bound','Lower bound','Pause'};

tic
for user_index=[5,8,9]
    user_fig_index=user_fig_index+1;
    instance_fig_index=0;
    data_path=[data_pwd,data_type{user_index}];
    for gesture_index = 9:9
        for position_index = 1:1
            for orientation_index = 1:1
                for instance_index = [1,2,4]
                    instance_fig_index=instance_fig_index+1;
                    data_file_name = [data_file_prefix,num2str(user_index), '-', num2str(gesture_index), '-', num2str(position_index),...
                        '-', num2str(orientation_index), '-', num2str(instance_index)];
                    disp(['Loading ', data_file_name])
                    try
                        % Generate Doppler Spectrum
                        [doppler_spectrum, freq_axis_bin,velocity_axis_bin] = generate_doppler_spectrum([data_path, data_file_name],...
                            n_receivers, n_antennas, n_subcarriers,'stft',wave_length,sample_rate,n_pca);
                        
                        [~,idx] = max(freq_axis_bin);
                        circle_length = length(freq_axis_bin) - idx;
                        doppler_spectrum = circshift(doppler_spectrum, [0 0 circle_length 0]);
                        freq_axis_bin=circshift(freq_axis_bin, [0 circle_length]);
                        velocity_axis_bin=circshift(velocity_axis_bin, [0 circle_length]);
                        
                        main_path_group_number = size(doppler_spectrum,4);
                        path_group_size=1;
                        
                        freq_bin_number=size(doppler_spectrum,3);
                        main_path_sum = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        velocity_bins_sum=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path_time = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        velocity_bins_time=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path_freq = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        velocity_bins_freq=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path_power = zeros(n_receivers,n_pca,freq_bin_number, main_path_group_number);
                        velocity_bins_power=zeros(n_receivers,n_pca,velocity_number,main_path_group_number);
                        
                        main_path = zeros(n_receivers,n_pca, freq_bin_number, main_path_group_number);
                        
                        
                        for kk = 2:2
                            for mm = 1:1
                                [main_path_sum(kk,mm,:,:),velocity_bins_sum(kk,mm,:,:)] = seam_carving_tailed_path(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,5,path_area,kernel_type,'v',velocity_bin,velocity_resolution,attenuation_efficient);
                                
                                [main_path_time(kk,mm,:,:),velocity_bins_time(kk,mm,:,:)] = seam_carving_time_tailed_path(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,2,path_area,kernel_type,'v',velocity_bin,velocity_resolution,attenuation_efficient);
                                
                                %
                                [main_path_freq(kk,mm,:,:),velocity_bins_freq(kk,mm,:,:)] = seam_carving_freq_tailed_path(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,2,100,kernel_type,'h',velocity_bin,velocity_resolution,attenuation_efficient);
                                
                                [main_path_power(kk,mm,:,:),velocity_bins_power(kk,mm,:,:)] = graph_matching_multipath_path(squeeze(doppler_spectrum(kk,mm,:,:)), velocity_axis_bin, main_path_group_number,path_group_size,1,2,velocity_bin,velocity_resolution,attenuation_efficient);
                                
                                
                                main_path(kk,mm,:,:)=main_path_freq(kk,mm,:,:)+main_path_time(kk,mm,:,:)+main_path_power(kk,mm,:,:);
                                
                                
                                h=zeros(4,1);
                                figure;
                                disp([data_file_name, '-r', num2str(kk),'-p', num2str(mm)])
                                colormap(jet);
                                mesh(1:size(doppler_spectrum,4),-60:60,squeeze(doppler_spectrum(kk,mm,:,:)));view([0,90]);
                                
                                if ~exist(save_path,'file')==0
                                    for line_index_h=1:size(plot_line_h,3)
                                        plot_data=squeeze(plot_line_h(user_fig_index,instance_fig_index,line_index_h,:,:));
                                        hold on;
                                        h(line_index_h)=plot3(plot_data(1,:),plot_data(2,:),plot_data(3,:)+0.1,'--','color',color_dot(line_index_h,:),'linewidth',5);
                                    end
                                    for line_index_v=1:size(plot_line_v,3)
                                        plot_data=squeeze(plot_line_v(user_fig_index,instance_fig_index,line_index_v,:,:));
                                        hold on;
                                        h(4)=plot3(plot_data(1,:),plot_data(2,:),plot_data(3,:)+0.1,'--','color',color_dot(4,:),'linewidth',5);
                                    end
                                else
                                    for line_index_h=1:size(plot_line_h,3)
                                        datacursormode on
                                        input('请描绘轨迹，并按回车继续：','s');
                                        curso_gcf=datacursormode(gcf);
                                        curso_info=getCursorInfo(curso_gcf);
                                        data=zeros(2,size(curso_info,2));
                                        for plot_index=1:size(curso_info,2)
                                            data(1,plot_index)=curso_info(plot_index).Position(1);
                                            data(2,plot_index)=curso_info(plot_index).Position(2);
                                            if(plot_index~=1 && data(1,plot_index)==data(1,plot_index-1))
                                                data(1,plot_index)=data(1,plot_index)+1;
                                            end
                                        end
                                        data=sortrows(data')';
                                        sequence=min(data(1,:)):(max(data(1,:))-min(data(1,:)))/size(plot_line_h,5):max(data(1,:));
                                        plot_line_h(user_fig_index,instance_fig_index,line_index_h,1,:)=sequence(1:size(plot_line_h,5));
                                        plot_line_h(user_fig_index,instance_fig_index,line_index_h,2,:)=interp1(data(1,:),data(2,:),  plot_line_h(user_fig_index,instance_fig_index,line_index_h,1,:));
                                        
                                        hold on;
                                        plot_data=squeeze(plot_line_h(user_fig_index,instance_fig_index,line_index_h,:,:));
                                        h(line_index_h)=plot3(plot_data(1,:),plot_data(2,:),plot_data(3,:)+0.1,':','color',color_dot(line_index_h,:),'linewidth',5);
                                        datacursormode off
                                    end
                                    
                                    for line_index_v=1:size(plot_line_v,3)
                                        datacursormode on
                                        curso_gcf=datacursormode(gcf);
                                        curso_info=getCursorInfo(curso_gcf);
                                        data=zeros(2,size(curso_info,2));
                                        for plot_index=1:size(curso_info,2)
                                            data(2,plot_index)=curso_info(plot_index).Position(1);
                                            data(1,plot_index)=curso_info(plot_index).Position(2);
                                            if(plot_index~=1 && data(1,plot_index)==data(1,plot_index-1))
                                                data(1,plot_index)=data(1,plot_index)+1;
                                            end
                                        end
                                        data=sortrows(data')';
                                        sequence=min(data(1,:)):(max(data(1,:))-min(data(1,:)))/size(plot_line_v,5):max(data(1,:));
                                        plot_line_v(user_fig_index,instance_fig_index,line_index_v,2,:)=sequence(1:size(plot_line_v,5));
                                        plot_line_v(user_fig_index,instance_fig_index,line_index_v,1,:)=interp1(data(1,:),data(2,:),  plot_line_v(user_fig_index,instance_fig_index,line_index_v,1,:));
                                        hold on;
                                        plot_data=squeeze(plot_line_v(user_fig_index,instance_fig_index,line_index_v,:,:));
                                        h(4)=plot3(plot_data(1,:),plot_data(2,:),plot_data(3,:)+0.1,':','color',color_dot(4,:),'linewidth',5);
                                        datacursormode off
                                    end
                                end
                                hold off;
                                set(gcf,'Position',get(0,'ScreenSize'));
                                lgd=legend(h,label_legend,'Location','northoutside','Orientation','horizontal');
                                xlim([0,size(doppler_spectrum,4)]);ylim([-60,60]);
                                
                                set (gca,'color','none', 'fontsize', 50); % fontsize
                                set(gca,'yTick',-60:30:60);
                                set(gca,'xTick',0:400:size(doppler_spectrum,4));
                                xlabel('Time (ms)','FontWeight','bold','FontSize',40); % x label
                                ylabel('Frequency Shift(Hz)','FontWeight','bold','FontSize',40); % y label
                    
                                saveas(gcf,[figures_dir,'fig2-',data_file_name, '.fig']);
                                saveas(gcf,[figures_dir,'fig2-',data_file_name, '.png']);
                                
                                %                                 saveas(gcf,[save_type,'-',data_file_name, '.fig']);
                                %                                 saveas(gcf,[save_type,'-',data_file_name, '.png']);
                                
                                %                                 figure;
                                %                                 colormap(jet);
                                %                                 mesh(1:size(main_path,4),squeeze(velocity_axis_bin),squeeze(main_path(kk,mm,:,:)));view([0,90]);
                                %                                 xlim([0,size(main_path,4)]);ylim([min(velocity_axis_bin),max(velocity_axis_bin)]);
                                %                                 set(gcf,'WindowStyle','normal','Position', [300,300,400,250]); % window size
                                %                                 set(gcf,'WindowStyle','normal','Position', [750+250*(instance_fig_index-1),300*(user_fig_index-1),300,250]); % window size
                                %                                 set (gca,'color','none', 'fontsize', 12); % fontsize
                                %                                 set(gca,'yTick',-3:0.5:3);
                                %                                 xlabel('Time (ms)'); % x label
                                %                                 ylabel('Velocity (m/s)'); % y label
                                %                                 colorbar; %Use colorbar only if necessary
                                %                                 caxis([min(main_path(:)),max(main_path(:))]);
                                %                                 saveas(gcf,[data_file_name, '-r', num2str(kk),'-p', num2str(mm),'-seam', '.png']);
                            end
                        end
                        
                    catch err
                        exception_fid = fopen([figures_dir, 'exception_log_',data_file_name,'.log'],'wt');
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
if ~exist(save_path,'file')~=0
    save(save_path,'plot_line_h', 'plot_line_v');
end
disp([save_type,'is finished'])
toc