clear 

load("array_sample.mat");
%----使用4种方法对309图做小鱼分割----%
k = 0;
%数据读入
I_309 = imread("309.bmp");
load("Mask_309.mat");
%训练模型并处理图像
Gray_norm(I_309,Mask,array_sample,k);
GrayRGB_norm(I_309,Mask,array_sample);
Histogram(I_309,Mask,array_sample);
Parzen_Window(I_309,Mask,array_sample);

%----使用灰度正态方法对其余4张图做小鱼分割----%
I_string = ["311.bmp" "313.bmp" "315.bmp" "317.bmp"];
Mask_string = ["Mask_311.jpg" "Mask_313.jpg" "Mask_315.jpg" "Mask_317.jpg"];
for k = 1:4
    I = imread(I_string(k));
    Mask = imread(Mask_string(k));
    Mask = im2double(Mask);
    %将Mask变换为规整的0,1
    for i = 1:240
        for j = 1:320
            if Mask(i,j) > 0.9
                Mask(i,j) = 1;
            else
                Mask(i,j) = 0;
            end
        end
    end
    Gray_norm(I,Mask,array_sample,k);
end