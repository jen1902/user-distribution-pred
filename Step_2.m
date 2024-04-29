clear all;
clc;

%% Load Data
% 200 UE
load('UEact_non_1.mat');
load('UEact_non_2.mat');
load('UEact_non_3.mat');
load('UEact_non_4.mat');
load('UEact_non_5.mat');
load('UEact_non_6.mat');
load('UEact_non_7.mat');
load('UEact_non_8.mat');
load('UEact_non_9.mat');
load('UEact_uni_1.mat');
load('UEgrid_non_1.mat');
load('UEgrid_non_2.mat');
load('UEgrid_non_3.mat');
load('UEgrid_non_4.mat');
load('UEgrid_non_5.mat');
load('UEgrid_non_6.mat');
load('UEgrid_non_7.mat');
load('UEgrid_non_8.mat');
load('UEgrid_non_9.mat');
load('UEgrid_uni_1.mat');
load('UEgrid_uni_2.mat');
load('UEgrid_uni_3.mat');
load('UEgrid_uni_4.mat');
load('UEgrid_uni_5.mat');
load('UEgrid_uni_6.mat');
load('UEgrid_uni_7.mat');
load('UEgrid_uni_8.mat');
load('UEgrid_uni_9.mat');


%% Combine Data
% Non Uniform
% UEpos_actual_non0715(1,:,:,:)=UEact_non_1(:,:,:);
% UEpos_actual_non0715(2,:,:,:)=UEact_non_2(:,:,:);
% UEpos_actual_non0715(3,:,:,:)=UEact_non_3(:,:,:);
% UEpos_actual_non0715(4,:,:,:)=UEact_non_4(:,:,:);
% UEpos_actual_non0715(5,:,:,:)=UEact_non_5(:,:,:);
% UEpos_actual_non0715(6,:,:,:)=UEact_non_6(:,:,:);
% UEpos_actual_non0715(7,:,:,:)=UEact_non_7(:,:,:);
% UEpos_actual_non0715(8,:,:,:)=UEact_non_8(:,:,:);
% UEpos_actual_non0715(9,:,:,:)=UEact_non_9(:,:,:);

% 
% save(['D:\Users\Desktop\UE_prediction\Step2_data\UEpos_actual_non0715'],['UEpos_actual_non0715']);

%% Deal with the divide
Size_new = size(UEgrid_non_1,1);
Size_part = 200/50;
% resizesize = 15;

for tttt = 1:Size_new
    for yyy = 1:Size_part
        for xxx = 1:Size_part               
            indexxx = (yyy-1)*Size_part+xxx;
            Min_1 = (xxx-1)*50+1;
            Max_1 = xxx*50;
            Min_2 = (yyy-1)*50+1;
            Max_2 = yyy*50;
            test = [Min_1 Max_1 Min_2 Max_2];
            UEgrid50_non_9_0715(tttt,:,indexxx,:,:) = UEgrid_non_9(tttt,:,Min_1:Max_1,Min_2:Max_2);
        end    
    end
end
save(['D:\Users\Desktop\UE_prediction\Step2_data\UEgrid50_non_9_0715'],['UEgrid50_non_9_0715']);

