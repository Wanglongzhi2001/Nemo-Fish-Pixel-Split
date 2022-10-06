function [] = Histogram(I,Mask,array_sample)
    I = im2double(I);
    %----预处理----%
    I_fish = I .* Mask;           %通过mask
%     figure
%     imshow(I_fish);
    I_gray = rgb2gray(I_fish);    %转为灰度图
    I_data(:,:,1) = I_gray;
    I_data(:,:,2:4) = I_fish;
%     figure
%     imshow(I_gray);
    
    m = 0;
    n = 0;
    for i = 1:length(array_sample)      %统计训练样本中两个分类的数量
        if array_sample(i,5) == 1
            m = m + 1;
        elseif array_sample(i,5) == -1
            n = n + 1;
        end
    end
    w1_sample = array_sample(1 : m, 1 : 4);
    w2_sample = array_sample(m + 1 : m + n, 1 : 4);
    figure
    title('第一类直方图')
    plotmatrix(w1_sample);
    figure
    title('第二类直方图')
    plotmatrix(w2_sample);
    
    Pw1 = zeros(50,50,50,50);
    Pw2 = zeros(50,50,50,50);
    
    for i = 1:m
        x = zeros(4,1);
        for k = 1:4
            if w1_sample(i,k) == 0
                x(k) = 1;
            else
                x(k) = ceil(w1_sample(i,k)/0.02);
            end
        end
        Pw1(x(1),x(2),x(3),x(4)) = Pw1(x(1),x(2),x(3),x(4)) + 1 / (0.02 ^ 4 * m);
    end
    
    for i = 1:n
        x = zeros(4,1);
        for k = 1:4
            if w2_sample(i,k) == 0
                x(k) = 1;
            else
                x(k) = ceil(w2_sample(i,k)/0.02);
            end
        end
        Pw2(x(1),x(2),x(3),x(4)) = Pw2(x(1),x(2),x(3),x(4)) + 1 / (0.02 ^ 4 * n);
    end
    
    pw1 = m / (m + n);                    %根据训练样本里两类数量的先验概率
    pw2 = 1 - pw1;
    pxi_w1 = zeros(240,320);
    pxi_w2 = zeros(240,320);
    I_cut = zeros(240,320,3);
    
    for i = 1:240
        for j = 1:320
            x = zeros(4,1);
            for k = 1:4
                if I_data(i,j,k) == 0
                    x(k) = 1;
                else
                    x(k) = ceil(I_data(i,j,k)/0.02);
                end
            end
            pxi_w1(i,j) = Pw1(x(1),x(2),x(3),x(4));
            pxi_w2(i,j) = Pw2(x(1),x(2),x(3),x(4));
            Pw1_xi = pxi_w1 * pw1;
            Pw2_xi = pxi_w2 * pw2;
            if Mask(i,j) == 0
                I_cut(i,j,:) = [0,0,0];
            elseif Pw1_xi(i,j) < Pw2_xi(i,j)
                I_cut(i,j,:) = [1,1,1];
            elseif Pw1_xi(i,j) == Pw2_xi(i,j)
                if(rand(1)<0.5)
                    I_cut(i,j,:) = [1,1,1];
                else
                    I_cut(i,j,:) = [0.8,0.5,0];
                end
            else
                I_cut(i,j,:) = [0.8,0.5,0];
            end
        end
    end
    imwrite(I_cut,"./results/Histogram_309.jpg");
    figure
    imshow(I_cut);
end