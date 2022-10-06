function [] = Parzen_Window(I,Mask,array_sample)
    I = im2double(I);
    %通过mask
    I_fish = I .* Mask;
%     figure
%     imshow(I_fish);
    %转为灰度图
    I_gray = rgb2gray(I_fish);
    I_data(:,:,1) = I_gray;
    I_data(:,:,2:4) = I_fish;
%     figure
%     imshow(I_gray);
    
    %统计训练样本中两个分类的数量
    m = 0;
    n = 0;
    for i = 1:length(array_sample)
        if array_sample(i,5) == 1
            m = m + 1;
        elseif array_sample(i,5) == -1
            n = n + 1;
        end
    end
    %整合训练样本
    w1_sample = array_sample(1 : m, 1 : 4);
    w2_sample = array_sample(m + 1 : m + n, 1 : 4);
    %画出训练样本的分布
%     figure
%     plotmatrix(w1_sample);
%     figure
%     plotmatrix(w2_sample);
    
    %创建概率密度查找表————4维数组，分度值设置为0.02，故每一维度有50种情况
    Pw1 = zeros(50,50,50,50);%第一类样本
    Pw2 = zeros(50,50,50,50);%第二类样本
    
    %d----窗大小（表示每一个样本会增加以它坐标为中心，边长为(2*d+1)的超立方体的概率密度
    d = 10;
    
    %对第一类的所有样本，得到第一类概率密度查找表Pw1
    for i = 1:m
        x = zeros(4,1);
        %计算x(4*1)——第i个样本所对应的4维坐标
        for k = 1:4
            if w1_sample(i,k) == 0
                x(k) = 1;
            else
                x(k) = ceil(w1_sample(i,k)/0.02);
            end
        end
        %给边长为(2*d+1)的超立方体增加概率密度
        for q = max(1,x(1) - d):min(50,x(1) + d)
            for w = max(1,x(2) - d):min(50,x(2) + d)
                for e = max(1,x(3) - d):min(50,x(3) + d)
                    for r = max(1,x(4) - d):min(50,x(4) + d)
                        Pw1(q,w,e,r) = Pw1(q,w,e,r) + 1 / (0.02 ^ 4 * m * (d * 2 + 1) ^ 4);
                    end
                end
            end
        end
    
    end
    
    %同理对第二类的所有样本，得到第一类概率密度查找表Pw2
    for i = 1:n
        x = zeros(4,1);
        for k = 1:4
            if w2_sample(i,k) == 0
                x(k) = 1;
            else
                x(k) = ceil(w2_sample(i,k)/0.02);
            end
        end
    
        for q = max(1,x(1) - 5):min(50,x(1) + 5)
            for w = max(1,x(2) - 5):min(50,x(2) + 5)
                for e = max(1,x(3) - 5):min(50,x(3) + 5)
                    for r = max(1,x(4) - 5):min(50,x(4) + 5)
                        Pw2(q,w,e,r) = Pw2(q,w,e,r) + 1 / (0.02 ^ 4 * m * 11 ^ 4);
                    end
                end
            end
        end
    end
    
    %根据训练样本里两类数量的先验概率
    pw1 = m / (m + n);
    pw2 = 1 - pw1;
    
    %创建后验概率和分类可视化图像
    pxi_w1 = zeros(240,320);
    pxi_w2 = zeros(240,320);
    I_cut = zeros(240,320,3);
    
    
    for i = 1:240
        for j = 1:320
            x = zeros(4,1);
            %计算x(4*1)——测试样本所对应的4维坐标
            for k = 1:4
                if I_data(i,j,k) == 0
                    x(k) = 1;
                else
                    x(k) = ceil(I_data(i,j,k)/0.02);
                end
            end
    
            %由坐标查找概率并计算后验概率
            pxi_w1(i,j) = Pw1(x(1),x(2),x(3),x(4));
            pxi_w2(i,j) = Pw2(x(1),x(2),x(3),x(4));
            Pw1_xi = pxi_w1 * pw1;
            Pw2_xi = pxi_w2 * pw2;
    
            %判决
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
    %输出图像
    imwrite(I_cut,"./results/Parzen_Window_309.jpg");
    figure
    imshow(I_cut);
end