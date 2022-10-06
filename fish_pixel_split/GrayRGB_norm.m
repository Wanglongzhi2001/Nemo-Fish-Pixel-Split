function [] = GrayRGB_norm(I,Mask,array_sample)
    I = im2double(I);
    I_fish = I .* Mask;           %通过mask
%     figure
%     imshow(I_fish);
    I_gray = rgb2gray(I_fish);    %转为灰度图
    I_data(:,:,1) = I_gray;
    I_data(:,:,2:4) = I_fish;
%     figure
%     imshow(I_gray);
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
    
    w1_sample = (array_sample((1:m),(1:4)));%读取训练样本并计算均值和方差
    w2_sample = (array_sample((m + 1:length(array_sample)),(1:4)));
    w1_avr = mean(w1_sample);
    w2_avr = mean(w2_sample);
    w1_cov = cov(w1_sample);
    w1_cov_det = abs(det(w1_cov));
    w1_cov_ni = w1_cov ^ (-1);
    w2_cov = cov(w2_sample);
    w2_cov_det = abs(det(w2_cov));
    w2_cov_ni = w2_cov ^ (-1);
        figure
    for i = 1:4
        X = 0:0.0001:1;                     %画出估计的两个类别正态分布pdf
        w1_norm = normpdf(X,w1_avr(i),w1_cov(i,i));
        w2_norm = normpdf(X,w2_avr(i),w2_cov(i,i));
        subplot(2,4,i)
        plot(X ,w1_norm,'r-');
        if i == 1
            title('第一类灰度值pdf')
        elseif i == 2
            title('第一类R分量pdf')
        elseif i == 3
            title('第一类G分量pdf')
        else
            title('第一类B分量pdf')
        end
        subplot(2,4,4 + i)
        plot(X ,w2_norm,'b-');
        if i == 1
            title('第二类灰度值pdf')
        elseif i == 2
            title('第二类R分量pdf')
        elseif i == 3
            title('第二类G分量pdf')
        else
            title('第二类B分量pdf')
        end
    end
    
    pw1 = m / (m + n);                    %根据训练样本里两类数量的先验概率
    % pw1 = 0.9;
    pw2 = 1 - pw1;
    
    I_cut = zeros(240,320,3);
    for i = 1:240
        for j = 1:320
            if Mask(i,j) == 0
                I_cut(i,j,:) = [0,0,0];
            else
                Pxk_w1 = 1 / sqrt(((2*pi) ^ 4) * w1_cov_det) * exp(-0.5 * (I_data(i,j) - w1_avr) * w1_cov_ni * (I_data(i,j) - w1_avr)');
                Pxk_w2 = 1 / sqrt(((2*pi) ^ 4) * w2_cov_det) * exp(-0.5 * (I_data(i,j) - w2_avr) * w2_cov_ni * (I_data(i,j) - w2_avr)');
                Pw1_xi = Pxk_w1 * pw1;
                Pw2_xi = Pxk_w2 * pw2;
                if Pw1_xi < Pw2_xi
                    I_cut(i,j,:) = [1,1,1];
                else
                    I_cut(i,j,:) = [0.8,0.5,0];
                end
            end
        end
    end
    imwrite(I_cut,"./results/GratRGB_norm_309.jpg");
    figure
    imshow(I_cut);
end