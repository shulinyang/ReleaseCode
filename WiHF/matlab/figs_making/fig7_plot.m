% MAIN
% Version 30-June-2019
% Help on http://liecn.github.com
clear;
clc;
close all;
tic

x = 1:6;
acc_g =[90,96,92,93,99,94];
acc_u = [90,96,92,93,99,94]+5;

far_g = [1,6,2,3,9,4]-0.5;
far_u = [1,6,2,3,9,4]-0.2;

fur_g = [1,6,2,3,9,4]+0.5;
fur_u = [1,6,2,3,9,4]+0.3;

[AX,H1,H2]=plotyy(x,[acc_g;acc_u],x,[far_g;far_u;fur_g;fur_u],'plot');

set(AX,'FontSize',30,'FontName','Times New Roman')%����x�ᡢ��y�ᡢ��y��̶��ֺź�����
set(AX(1),'Xcolor','k','Ycolor','k')%����x�ᡢ��y��̶�����Ϊ��ɫ��
set(AX(2),'Xcolor','k','Ycolor','k')%����x�ᡢ��y��̶�����Ϊ��ɫ��
set(AX,'Xlim',[0,7],'xtick',1:6)%����x�����ݷ�Χ��207.5��217.1�����̶���ʾ��208��209,210...217��
set(AX(1),'ylim',[85,100],'ytick',85:3:100)%������y�����ݷ�Χ��0��0.5�����̶���ʾ��0,0.1,0.2...0.5��
set(AX(2),'ylim',[0,15],'ytick',0:3:15)%������y�����ݷ�Χ��0��3�����̶���ʾ��0,1,2,3��
set(H1(1),'Linestyle','-','color','b','Linewidth',2.5);%���õ�һ�����ߵ����͡���ɫ����ϸ
set(H1(2),'Linestyle','-.','color','b','Linewidth',2.5);%���õ�һ�����ߵ����͡���ɫ����ϸ
% line(x,y2,'linestyle','-','color','r','Linewidth',2.5,'parent',AX(1));%�ڵ�һ���������ٻ�һ�����ߡ�
set(H2(1),'Linestyle','-','color','r','Linewidth',2.5);%���õڶ������ߵ����͡���ɫ����ϸ
set(H2(2),'Linestyle','-.','color','r','Linewidth',2.5);%���õڶ������ߵ����͡���ɫ����ϸ
set(H2(3),'Linestyle','-','color','g','Linewidth',2.5);%���õڶ������ߵ����͡���ɫ����ϸ
set(H2(4),'Linestyle','-.','color','g','Linewidth',2.5);%���õڶ������ߵ����͡���ɫ����ϸ
set(get(AX(1),'Ylabel'),'string','Acc Percent (%)','FontWeight','bold','FontSize',30);%������y������ֺ�����
set(get(AX(2),'Ylabel'),'string','FAR and FUR (%)','FontWeight','bold','FontSize',30);%������y������ֺ�����
set(get(AX(1),'Xlabel'),'string','Gesture/User Index','FontWeight','bold','FontSize',30);%����x������������С������
legend(H1,{'Acc-Ges','ACC-Usr'},'Orientation','horizontal')
legend(H2(1:2),{'FAR-Ges','FAR-Usr'},'Orientation','horizontal')
legend(H2(3:4),{'FUR-Ges','FUR-Usr'},'Orientation','horizontal')

set(gcf,'Position',get(0,'ScreenSize'));
%     lgd=legend(h,label_legend,'Location','eastoutside');


% set (gca,'color','none', 'fontsize', 50); % fontsize

% saveas(gcf,[figures_dir,save_type,'-',data_file_name, '.fig']);
% saveas(gcf,[figures_dir,save_type,'-',data_file_name, '.png']);
toc