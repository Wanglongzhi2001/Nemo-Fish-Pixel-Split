function [] = Gray_norm(I,Mask,array_sample,k)
    I = im2double(I);
    I_fish = I .* Mask;           %通过mask
    figure
    subplot(1,3,1)
    imshow(I_fish);
    I_gray = rgb2gray(I_fish);    %转为灰度图
    subplot(1,3,2)
    imshow(I_gray);
    %----建立模型----%
    m = 0;
    n = 0;
    for i = 1:length(array_sample)      %统计训练样本中两个分类的数量
        if array_sample(i,5) == 1
            m = m + 1;
        elseif array_sample(i,5) == -1
            n = n + 1;
        end
    end
    w1_sample = (array_sample((1:m),:));%读取训练样本并计算均值和方差
    w2_sample = (array_sample((m + 1:length(array_sample)),:));
    mean1_gray = mean(w1_sample(:,1));
    var1_gray = var(w1_sample(:,1));
    mean2_gray = mean(w2_sample(:,1));
    var2_gray = var(w2_sample(:,1));
    
    
    X = 0:0.0001:1;                     %画出估计的两个类别正态分布pdf
    w1_norm = normpdf(X,mean1_gray,var1_gray);
    w2_norm = normpdf(X,mean2_gray,var2_gray);
    
    pw1 = m / (m + n);                    %根据训练样本里两类数量的先验概率
    pw2 = 1 - pw1;
    
    %计算后验概率
    Pxi_w1 = 1 / sqrt(2 * pi * var1_gray) * exp( - (I_gray - mean1_gray).^ 2 / (2 * var1_gray));
    Pxi_w2 = 1 / sqrt(2 * pi * var2_gray) * exp( - (I_gray - mean2_gray).^ 2 / (2 * var2_gray));
    
    Pw1_xi = Pxi_w1 .* pw1; 
    Pw2_xi = Pxi_w2 .* pw2; 
    
    %----比较后验概率并作出决策----%
    I_cut = zeros(240,320,3);
    for i = 1:240
        for j = 1:320
            if Mask(i,j) == 0
                I_cut(i,j,:) = [0,0,0];
            elseif Pw1_xi(i,j) < Pw2_xi(i,j)
                I_cut(i,j,:) = [1,1,1];
            else
                I_cut(i,j,:) = [0.8,0.5,0];
            end
        end
    end
    subplot(1,3,3)
    imshow(I_cut);
    img_name = ["./results/Gray_norm_309.jpg" "./results/Gray_norm_311.jpg" "./results/Gray_norm_315.jpg" "./results/Gray_norm_317.jpg" "./results/Gray_norm_319.jpg"];
    imwrite(I_cut,img_name(k+1));
    if k == 0
        figure
        subplot(2,1,1)
        plot(X ,w1_norm,'r-');title('第一类灰度值pdf')
        subplot(2,1,2)
        plot(X ,w2_norm,'b-');title('第二类灰度值pdf')
    end
end