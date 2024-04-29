%% Load data
clear all;
clc;
load('CNNans.mat');
load('InputCNN.mat');
load('ResultCNN.mat'); %(Batch*16)*3*50*50
Len = 1500;
Wid = 1500;
size1 = 50;
size2 = 50;
size11 = 200;
size22 = 200;
%% Reshape data
Batchsize = size(ResultCNN,1);
CNN_Ans = reshape(CNNans,[Batchsize/16,16,3,50,50]);
CNN_Input = reshape(InputCNN,[Batchsize/16,16,3,50,50]);
CNN_Pred = reshape(ResultCNN,[Batchsize/16,16,3,50,50]);
Time_taken = 3;
CNN_Ans_take = CNN_Ans(:,:,Time_taken,:,:);
CNN_Inputs_take = CNN_Input(:,:,Time_taken-2,:,:);
CNN_Pred_take = CNN_Pred(:,:,Time_taken,:,:);
CNN_Ans_take = reshape(CNN_Ans_take,[Batchsize/16,16,50,50]);
CNN_Inputs_take = reshape(CNN_Inputs_take,[Batchsize/16,16,50,50]);
CNN_Pred_take = reshape(CNN_Pred_take,[Batchsize/16,16,50,50]);
threh = 0.4;
for bbbb = 1:Batchsize/16
    All_xCNNans = [];
    All_yCNNans = [];
    CNN_pos_ans = [];
    All_xCNNpred = [];
    All_yCNNpred = [];
    CNN_pos_pred = []; 
    Wrongratio_all = [];
    CNNans_now = [];
    CNNpred_now = [];
    CNNinput_now = [];
    All_xCNNinput =[];
    All_yCNNinput =[];
    CNNans_now(:,:,:) = CNN_Ans_take(bbbb,:,:,:) ;
    CNNpred_now(:,:,:) = CNN_Pred_take(bbbb,:,:,:) ;
    CNNinput_now(:,:,:) = CNN_Inputs_take(bbbb,:,:,:) ;
    for yyy = 1:4
        for xxx = 1:4
            indexxx = (yyy-1)*4+xxx;
            Min_1 = (xxx-1)*50+1;
            Max_1 = xxx*50;
            Min_2 = (yyy-1)*50+1;
            Max_2 = yyy*50;
            test = [Min_1 Max_1 Min_2 Max_2];
            CNNans_comb(Min_1:Max_1,Min_2:Max_2) = CNNans_now(indexxx,:,:);
            CNNpred_comb(Min_1:Max_1,Min_2:Max_2) = CNNpred_now(indexxx,:,:);
            CNNinput_comb(Min_1:Max_1,Min_2:Max_2) = CNNinput_now(indexxx,:,:);
        end    
    end
    CNNpred_Bi = CNNpred_comb;
%     CNNpred_all_re = CNNpred_all;
%     ZeroMatrix = ones(212,212);
%     ZeroMatrix(7:206,7:206) = CNNpred_all_re;
%     CNNpred_all_re = ZeroMatrix;
%     cutidx = [];
%     for m = 7:206
%         for n = 7:206
%             cutidx = [cutidx,[n;m]];
%         end
%     end
%     [CNNpred_all_re, THESHOLD] = CA_CFAR(CNNpred_all_re,cutidx);
%     CNNpred_all_re = reshape(CNNpred_all_re,200,200);
    for aa = 1:size11
        for bb = 1:size22
            if CNNpred_comb(aa,bb) >= threh
                CNNpred_Bi(aa,bb) = 1;
            else
                CNNpred_Bi(aa,bb) = 0;
            end
        end
    end
    UENum = find(CNNpred_Bi~=0);
    [rowCNN,colCNN,dimCNN]= find(CNNans_comb~=CNNpred_Bi);
    LenCNN = length(rowCNN);
    Allsize= size11*size22;
    Wrongratio = LenCNN/Allsize;
    Wrongratio_all = [Wrongratio_all ; Wrongratio];
    CNNpred_Bi = fliplr(CNNpred_Bi);
    CNNans_comb = fliplr(CNNans_comb);
    CNNinput_comb = fliplr(CNNinput_comb);
    for aaa = 1:size11
        for bbb = 1:size22
            UENumm_CNNans = CNNans_comb(aaa,bbb);
            if UENumm_CNNans ~=0
                PosX_CNNans = rand(UENumm_CNNans,1).*Wid/size22 + (bbb-1).*Wid/size11;
                All_xCNNans = [All_xCNNans ; PosX_CNNans];
                PosY_CNNans = rand(UENumm_CNNans,1).*Len/size11 + (aaa-1).*Len/size11;
                All_yCNNans = [All_yCNNans ; PosY_CNNans];
                CNN_pos_ans = [All_xCNNans All_yCNNans];                
            end
            UENumm_CNNpred = CNNpred_Bi(aaa,bbb);
            if UENumm_CNNpred ~=0
                PosX_CNNpred = rand(UENumm_CNNpred,1).*Wid/size22 + (bbb-1).*Wid/size11;
                All_xCNNpred = [All_xCNNpred ; PosX_CNNpred];
                PosY_CNNpred = rand(UENumm_CNNpred,1).*Len/size11 + (aaa-1).*Len/size11;
                All_yCNNpred = [All_yCNNpred ; PosY_CNNpred];
                CNN_pos_pred = [All_xCNNpred All_yCNNpred];                
            end
            UENumm_CNNinput = CNNinput_comb(aaa,bbb);
            if UENumm_CNNinput ~=0
                PosX_CNNinput = rand(UENumm_CNNinput,1).*Wid/size22 + (bbb-1).*Wid/size11;
                All_xCNNinput = [All_xCNNinput ; PosX_CNNinput];
                PosY_CNNinput = rand(UENumm_CNNinput,1).*Len/size11 + (aaa-1).*Len/size11;
                All_yCNNinput = [All_yCNNinput ; PosY_CNNinput];
                CNN_pos_input = [All_xCNNinput All_yCNNinput];                
            end

        end
    end

    Len_pred = length(CNN_pos_pred(:,1));
    Len_ans = length(CNN_pos_ans(:,1));
    Len_input = length(CNN_pos_input(:,1));
    All_pred = [];
    All_ans = [];
    All_input = [];
    All_pred=  [CNN_pos_pred ; zeros(300-Len_pred,2)];
    All_ans = [CNN_pos_ans ; zeros(300-Len_ans,2)];
    All_input = [CNN_pos_input ; zeros(300-Len_input,2)];
    
    CNNpred(bbbb,:,:) = All_pred;
    CNNgt(bbbb,:,:) = All_ans;
    CNNinput(bbbb,:,:) = All_input;
%     figure(5555555);
%     plot(CNN_pos_pred(:,1),CNN_pos_pred(:,2),'bx');
%     hold on;
%     plot(CNN_pos_ans(:,1),CNN_pos_ans(:,2),'ro');
%     hold on;
%     plot(CNN_pos_input (:,1),CNN_pos_input (:,2),'g.');
    
end

Comp_mean0730 = mean(Wrongratio_all,1);
save(['D:\Users\Desktop\Code\Prediction\CNNpred'],['CNNpred']);
save(['D:\Users\Desktop\Code\Prediction\CNNgt'],['CNNgt']);
save(['D:\Users\Desktop\Code\Prediction\CNNinput'],['CNNinput']);
save(['D:\Users\Desktop\Code\Prediction\Comp_mean'],['Comp_mean']);

