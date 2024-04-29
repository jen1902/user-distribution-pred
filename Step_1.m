clear all;
clc;

%% Data Production
Len = 1500 ; %m
Wid = 1500 ; %m
size11 = 5;
size22 = 5;

%% UE Grid Production
X_pergrid = Len / size11;
Y_pergrid = Wid / size22;
Matrix_1 = [4 3 1 1 1 ;5 3 1 1 1; 5 2 1 1 2; 4 3 1 1 2; 1 1 1 1 2];
Matrix_2 = [2 2 2 2 2;2 3 2 2 2; 2 2 3 2 2;2 2 2 2 1;2 2 2 2 2];

Unit = sum(sum(Matrix_1));
Density = 4;%4
LenDen = length(Density);
UENum = [];
for ddd = 1:LenDen
    UEDist_1 = Density(ddd).*Matrix_1; 
    UEDist_2 = Density(ddd).*Matrix_2;
    UEDist_all_1(ddd,:,:)= UEDist_1(:,:);
    UEDist_all_2(ddd,:,:)= UEDist_2(:,:);
end


%% Data Production
% Basic Paramenters
alpha = [0:0.1:1] ;
Scen_size = 200;
Vel_size = 20;
UENum = 50*Density; %200
size1_new = 200;
size2_new = 200;
Step_size = 20;

% Velocity Parameters
Velocity = 3.6*1000/3600; %m/s2
Velocity_val = 0.2*1000/3600;
for bbbb = 1:Scen_size
    
    %% Initialization for each batch
    UE_1 = [];
    UEInit_pos = [];
    for xxx = 1:size11
        for yyy = 1:size22
            xx_temp = 6-xxx ;         
            Num_pergrid1 = UEDist_all_1(ddd,xx_temp,yyy);
            UE_x1 = X_pergrid.*rand(Num_pergrid1,1)+ (yyy-1)*X_pergrid;
            UE_y1 = Y_pergrid.*rand(Num_pergrid1,1)+ (xxx-1)*Y_pergrid;
            UE_1_temp = [UE_x1 UE_y1];
            UE_1 = [UE_1;UE_1_temp];
        end
    end
    UEInit_pos = UE_1;
    for aa = 1:size1_new
        for bb = 1:size2_new
            Xmin = 0+(bb-1)*Len/size2_new;
            Xmax = 0+bb*Len/size2_new;
            Ymin = 0+(aa-1)*Len/size2_new;
            Ymax = 0+aa*Len/size2_new;
            tempnum = find(Xmin<=UEInit_pos(:,1) & UEInit_pos(:,1)<Xmax & Ymin<=UEInit_pos(:,2) & UEInit_pos(:,2)<Ymax);
            Num_grid = length(tempnum);
            All_grid_init(aa,bb) =  Num_grid;
        end
    end
    
    UEgrid_non_9(bbbb,1,:,:) =All_grid_init;
    UEact_non_9(bbbb,1,:,:) = UEInit_pos;
    
    
    %% For Different Velocity
    Angle = randi([0,360],UENum,1);
    Speed_xNew = Velocity.*cosd(Angle).*Step_size;
    Speed_yNew = Velocity.*sind(Angle).*Step_size;
    Angle_val = randi([0,360],UENum,1);
    delta_x = Velocity_val.*cosd(Angle_val).*Step_size;
    delta_y = Velocity_val.*sind(Angle_val).*Step_size;
    alpha_index = randi([1,10],UENum,1);    
    record = 1;
    for vvv = 2:(Vel_size+1)
        alpha_now = alpha(alpha_index);
        alpha_now = alpha_now.';
        % Update position
        UE_newpos_x = UEInit_pos(:,1) + Speed_xNew ;
        UE_newpos_y = UEInit_pos(:,2) + Speed_yNew ; 
        UE_newpos = [UE_newpos_x  UE_newpos_y];
        % Update speed
        Speed_xpre = Speed_xNew;
        Speed_ypre = Speed_yNew;
        w = randn(2,1);
        y = conv(w,ones(1,100)/10,'same');
        Speed_xNew = Speed_xpre.*alpha_now + (1-alpha_now).*delta_x + sqrt(1-alpha_now.^2).*y(1,1)*5;
        Speed_yNew = Speed_ypre.*alpha_now + (1-alpha_now).*delta_y + sqrt(1-alpha_now.^2).*y(2,1)*5;
        % Grid record
        for aa = 1:size1_new
            for bb = 1:size2_new
                Xmin = 0+(bb-1)*Len/size2_new;
                Xmax = 0+bb*Len/size2_new;
                Ymin = 0+(aa-1)*Len/size2_new;
                Ymax = 0+aa*Len/size2_new;
                tempnum = find(Xmin<=UE_newpos_x & UE_newpos_x<Xmax & Ymin<=UE_newpos_y & UE_newpos_y<Ymax);
                Num_grid = length(tempnum);
                All_grid(aa,bb) =  Num_grid;
            end
        end

        
        UEInit_pos = UE_newpos;        
        UEgrid_non_9(bbbb,record+1,:,:) =All_grid;
        UEact_non_9(bbbb,record+1,:,:) = UE_newpos;
        record = record+1;
%         figure(123456);
%         imagesc(All_grid);
%         pause(0.5)
    end
%      AAA = []; 
%      AAA(:,:) = UEact_non_1(1,3,:,:);
%      BBB = [];
%      BBB(:,:) = UEact_non_1(1,8,:,:);
%      figure(123456789);
%      plot(AAA(:,1),AAA(:,2),'bx');
%      hold on;
%      plot(BBB(:,1),BBB(:,2),'rx');
    
end
save(['D:\Users\Desktop\UE_prediction\Step1_data\UEgrid_non_9'],['UEgrid_non_9']);
save(['D:\Users\Desktop\UE_prediction\Step1_data\UEact_non_9'],['UEact_non_9']);
