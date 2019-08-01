% MAIN
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;
data_root = 'F:\wf_tally/';
save_type='fig2';

figures_dir = [data_root,'FIGURES/',save_type,'/'];

save_path=[figures_dir,'data.mat'];

if ~exist(save_path,'file')==0
    plot_line_h=cell2mat(struct2cell(load(save_path, 'plot_line_h')));
    plot_line_v=cell2mat(struct2cell(load(save_path, 'plot_line_v')));
else
    disp(['No data, run ',save_type,'_generate.m first! ']);
    exit;
end

N = 5;
color_dot = [ 0 0 0;0.6769,0.4447,0.7114;0.9153,0.2816,0.2878;1,0.75,0.79];
%                                 1,0.75,0.79;1.0000,0.5984,0.2000;
label_legend={'Instance#1','Instance#2','Instance#4'};
linestyle_list={'-',':','-.'};
tic
h=zeros(3,1);
user_list = [5,8,9];
instance_list = [1,2,4];
for user_fig_index=1:size(plot_line_h,1)
    figure;
    for instance_fig_index=1:size(plot_line_h,2)
        data_file_name = [num2str(user_list(user_fig_index)), '-9-1-1-', num2str(instance_list(instance_fig_index))];
        
        plot_data=squeeze(plot_line_h(user_fig_index,instance_fig_index,1,:,:));
        plot_data_2=squeeze(plot_line_h(user_fig_index,instance_fig_index,2,:,:));
        plot_data_3=squeeze(plot_line_h(user_fig_index,instance_fig_index,3,:,:));
        
        hold on;
        h(instance_fig_index)=fill([plot_data_2(1,:),fliplr(plot_data_3(1,:))],[plot_data_2(2,:),fliplr(plot_data_3(2,:))],color_dot(instance_fig_index,:));
        alpha(0.5);
        for line_index_v=1:size(plot_line_v,3)
            plot_data=squeeze(plot_line_v(user_fig_index,instance_fig_index,line_index_v,:,:));
            hold on;
            plot(plot_data(1,:),plot_data(2,:),linestyle_list{instance_fig_index},'color',color_dot(instance_fig_index,:),'linewidth',3);
        end
    end
    hold off;
    set(gcf,'Position',get(0,'ScreenSize'));
%     lgd=legend(h,label_legend,'Location','eastoutside');
    xlim([0,1700]);ylim([-60,60]);
    
    set (gca,'color','none', 'fontsize', 50); % fontsize
    set(gca,'yTick',-60:30:60);
    set(gca,'xTick',0:400:1700);
    xlabel('Time (ms)','FontWeight','bold','FontSize',40); % x label
    ylabel('Frequency Shift(Hz)','FontWeight','bold','FontSize',40); % y label
    saveas(gcf,[figures_dir,save_type,'-',data_file_name, '.fig']);
    saveas(gcf,[figures_dir,save_type,'-',data_file_name, '.png']);
    %     saveas(gcf,[save_type,'-',data_file_name, '.fig']);
    %     saveas(gcf,[save_type,'-',data_file_name, '.png']);
end
disp([save_type,'is finished'])
toc