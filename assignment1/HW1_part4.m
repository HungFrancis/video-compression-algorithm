% 5LSE0 pratical assignment part 1
% Author: Shao Hsuan Hung (s.hung@student.tue.nl)
% ID: 1723219
% Update date: 1/7/2023
% MATLAB version: R2022a
% Required file: lena.pgm, peppers.pgm, cameraman.pgm, butterfly.pgm
clc;clear;close all
set(0,'defaultfigurecolor','w');
%% 4.a, Generate 8x8 Haar matrix
haar_matrix = Haar_Matrix(8);
%% 4.b Implement 8 point haar transformation & inverse transformation
clear;
lena = double(imread('butterfly.pgm'));
img(:,:,:)  = zeros(size(lena,1),size(lena,2),8);
snr_array = zeros(8,1);
for i = 1:1:8
    img(:,:,i) = normalized(HaarTransform(lena,8,i));
    figure
    imshow(img(:,:,i),[])
end
% Save the result 
% imwrite(img(:,:,1),'bf1_HT.png');
% imwrite(img(:,:,2),'bf2_HT.png');
% imwrite(img(:,:,3),'bf3_HT.png');
% imwrite(img(:,:,4),'bf4_HT.png');
% imwrite(img(:,:,5),'bf5_HT.png');
% imwrite(img(:,:,6),'bf6_HT.png');
% imwrite(img(:,:,7),'bf7_HT.png');
% imwrite(img(:,:,8),'bf8_HT.png');
%% 4.c PDF diagram of coefficient (7,8) & (3,2) for both DCT and haar transformation 
cameraman = double(imread('cameraman.pgm'));
haar_8_matrix = Haar_Matrix(8);
% Get the haar tranform coefficient and get the assigned component
haar_transform  = @(blockstruc) haar_8_matrix*blockstruc.data*haar_8_matrix';
inv_haar_transform = @(blockstruc) haar_8_matrix'*blockstruc.data*haar_8_matrix;
transformed_img = blockproc(cameraman,[8 8],haar_transform);
get_32_matrix = [0 0 0 0 0 0 0 0 ;
          0 0 1 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;];
Get32 = @(blockstruct) blockstruct.data.*get_32_matrix;
filtered = blockproc(transformed_img,[8 8],Get32);
haar_component32 = nonzeros(filtered);
get_78_matrix = [0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 0 0 ;
          0 0 0 0 0 0 1 0 ;];
Get78 = @(blockstruct) blockstruct.data.*get_78_matrix;
filter = blockproc(transformed_img,[8 8],Get78);
haar_component78 = nonzeros(filter);
% DCT
dct_block = dctmtx(8);
dct = @(block_struct) dct_block*(block_struct.data)*dct_block';
dct_img = (blockproc(cameraman,[8 8],dct));
DCT_component32 = nonzeros(blockproc(dct_img,[8 8],Get32));
DCT_component78 = nonzeros(blockproc(dct_img,[8 8],Get78));
A = blockproc(dct_img,[8 8],Get78);
figure
pd = fitdist(haar_component32,'kernel');
x_values1 = min(haar_component32):0.01:max(haar_component32) ; 
y_values1 = pdf(pd,x_values1);
plot(x_values1,y_values1,'b-','DisplayName','Haar Transform')
hold on; grid on;
pd = fitdist(DCT_component32,'kernel');
x_values2 = min(DCT_component32):0.01:max(DCT_component32) ; 
y_values2 = pdf(pd,x_values2);
plot(x_values2,y_values2,'r-','DisplayName','DCT Transform')
xlim([-10 10])
ylim([0 0.25])
xlabel("Coefficient");
ylabel("PDF");
title("PDF diagram coefficients (3,2) (low-frequency)")
legend

figure
pd = fitdist(haar_component78,'kernel');
x_values3 = min(haar_component78):0.01:max(haar_component78) ; 
y_values3 = pdf(pd,x_values3);
plot(x_values3,y_values3,'b-','DisplayName','Haar Transform')
xlim([-10 10])
ylim([0 0.25])
hold on; grid on;
pd = fitdist(DCT_component78,'kernel');
x_values4 = min(DCT_component78):0.01:max(DCT_component78); 
y_values4 = pdf(pd,x_values4);
plot(x_values4,y_values4,'r-','DisplayName','DCT Transform');
xlabel("Coefficient");
ylabel("PDF");
title("PDF diagram of coefficients (7,8) (high-frequency)")
legend
%% 4.d
cameraman = double(imread('cameraman.pgm'));
butterfly = double(imread('butterfly.pgm'));
% Haar
bpp = 8;
haar_tranform_man(:,:,:) = zeros(size(cameraman,1),size(cameraman,2),bpp);
haar_tranform_fly(:,:,:) = zeros(size(butterfly,1),size(butterfly,1),bpp);
haar_snr_array_man = zeros(bpp,1);
haar_snr_array_fly = zeros(bpp,1);
for i = 1:bpp
    [haar_snr_array_man(i),haar_tranform_man(:,:,i)] = HaarTransform(cameraman,8,i);
    [haar_snr_array_fly(i),haar_tranform_fly(:,:,i)] = HaarTransform(butterfly,8,i);
end
% DCT
DCT_transform_man(:,:,:)= zeros(size(cameraman,1),size(cameraman,2),bpp);
DCT_tranform_fly(:,:,:) = zeros(size(butterfly,1),size(butterfly,1),bpp);
DCT_snr_array_man = zeros(bpp,1);
DCT_snr_array_fly = zeros(bpp,1);
for i = 1:bpp
    [DCT_snr_array_man(i),DCT_transform_man(:,:,i)] = DCT_pipeline1(cameraman,i);
    [DCT_snr_array_fly(i), DCT_tranform_fly(:,:,i)] = DCT_pipeline1(butterfly,i);
end

figure
plot(haar_snr_array_man,"--o","DisplayName","Haar Transform");

hold on; grid on;
plot(DCT_snr_array_man,"--*","DisplayName","DCT Transform");

title("SNR versus the bitrate for different tranformation on cameraman.pgm")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend

figure
plot(haar_snr_array_fly,"--o","DisplayName","Haar Transform");
hold on; grid on;
plot(DCT_snr_array_fly,"--*","DisplayName","DCT Transform");
title("SNR versus the bitrate for different tranformation on butterfly.pgm")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend
%% ------------------- Function -------------------------%%
function [snr_value, compressed_img] = HaarTransform(img,N,bpp)
    % Input parameters:
        % img: the image that you want to compress
        % N: the point to transform
        % bpp : bit per pixel for quantization
    % Step1, Generate the Haar matrix by NxN
    haar = Haar_Matrix(N);
    % Step2. Haar transform
    haar_transform  = @(blockstruc) haar*blockstruc.data*haar';
    inv_haar_transform = @(blockstruc) haar'*blockstruc.data*haar;
    transformed_img = blockproc(img,[8 8],haar_transform);
    % Step3. Quantized the harr coefficient
    [haar_quant_img, q_level, delta] = Haar_quantization(transformed_img,bpp);
    % Step4. Encoding the haar coefficient
    [decode_dict,code] = encoding(haar_quant_img,q_level);
    % Step5. Decoding the haar coefficient
    haar_quant_img = decoding(code,decode_dict,size(img,2));
    % Step6. Dequantized the haar coefficient
    haar_dequant_img = Haar_dequantization(haar_quant_img,delta);
    % Step7. inverse haar transform
    compressed_img = blockproc(haar_quant_img,[8 8],inv_haar_transform);
    %
    snr_value = calculate_snr(img,compressed_img);
end
%% Generate the nxn haar matrix
function haar_matrix = Haar_Matrix(N)
    % Check the input argument
    if ((log2(N)-floor(log2(N)))~=0 || N<2)
        error('The input argument should be of form 2^k');
    end
    n = log2(N);
    % Build the p q matrix 2^p + q -1 = k
    p(1:2)= [0 0]; q(1:2) = [0 1];
    for i = 1:n-1
        p= [p i*ones(1,2^i)];
        q= [q   1:2^(i)];
    end
    haar_matrix = zeros(N); haar_matrix(1,:) = 1;
    for i = 2:(N)
        P = p(i);
        Q = q(i);
        for j = N*(Q-1)/(2^P):N*(Q-0.5)/(2^P)-1
            haar_matrix(i,j+1) = 2^(P/2);
        end
        for j = N*(Q-0.5)/(2^P):N*(Q)/(2^P)-1
            haar_matrix(i,j+1) = -2^(P/2);
        end
    end
    %
    haar_matrix = haar_matrix/sqrt(N);
end
%%
function [new_img,q_level,delta] = Haar_quantization(img,bit_rate)
    % Return:
    %   q_level: for encoding 
    %   delta  : for dequantization
    MaxValue = max(img(:));
    MinValue = min(img(:));
    delta = (max(img(:))-min(img(:)))/(2^bit_rate);
    new_img = (fix(img./delta));
    q_level = unique(new_img);
    maxValue =  max(max(img));
    minValue = min(min(img));
    delta = (maxValue - minValue)/(2^bit_rate);
    q_level = zeros(bit_rate,1);
    for i =1:1:2^bit_rate
        q_level(i) =  minValue + delta*(i-1);
    end
    new_img = (fix(img./delta));
     q_level = unique(new_img);
end
function new_img = Haar_dequantization(quanted_img,delta)
    new_img = quanted_img.*delta+delta/2;
%         de_img = img*delta+delta/2;
end
%% Implement the SNR
function snr = calculate_snr(ori_img, compress_img)
    mse = 0;
    square = 0;
    ori_img = normalized(ori_img);
    compress_img = normalized(compress_img);
    for i = 1:1:size(compress_img,1)
        for j = 1:1:size(compress_img,2)
                square = square + (compress_img(i,j))^2;
                mse = mse + (ori_img(i,j)-compress_img(i,j))^2;
        end
    end
    snr = 10*log10(square/mse);
end
function new_img = normalized(img)
    new_img = (img-min(img(:)))./(max(img(:)-min(img(:))));
end

%%
function [snr_value, idct_img] = DCT_pipeline1(img,bpp)
    % Step 1.DCT
    dct_block = dctmtx(8);
    dct = @(block_struct) dct_block*(block_struct.data)*dct_block';
    dct_img = (blockproc(img,[8 8],dct));
    % Step 2. Quantized coeff.
    quant_matrix = ones(8,8); % Uniform quantize matrix
    quant = @(block_strct) (block_strct.data ./ quant_matrix);
    quant_dct_img = blockproc(dct_img,[8 8],quant);
    [q_level,q_img,delta] = DCT_quantize(quant_dct_img,bpp);
    % Step3. Dequantize
    dq_img = DCT_dequantize(q_img,delta);
    dquant = @(block_struct) (block_struct.data .* quant_matrix);
    dq_img_quant = blockproc(q_img,[8 8],dquant);
    % Step4. Inverse DCT
    idct = @(block_struct) dct_block'*(block_struct.data)*dct_block;
    idct_img = blockproc(dq_img_quant,[8 8],idct);
%     % Step5. rescale the idct image 
    snr_value = calculate_snr(img,idct_img);
    error = img - idct_img;
%     snr_value = 10*log10(var(img,1,'all')/var(error,1,'all'));
end
%% Quantizer
function [q_level, new_img, delta] = DCT_quantize(img,bit_rate)
    maxValue =  max(max(img));
    minValue = min(min(img));
    delta = (maxValue - minValue)/(2^bit_rate);
    q_level = zeros(bit_rate,1);
    for i =1:1:2^bit_rate
        q_level(i) =  minValue + delta*(i-1);
    end
    new_img = (fix(img./delta));
end
function de_img = DCT_dequantize(img,delta)
    de_img = img*delta+delta/2;
end
%% Encoding 
function [dict,code] = encoding(img,q_level)
    img = reshape(img,[],1);
    q_level = reshape(q_level,[],1);
    probs = zeros(size(q_level,1),1);
    for k = 1:1:size(q_level,1)
        number = 0;
        for i = 1:1:size(img,1) % Calculate the frequency of quantized value
            if(img(i)==q_level(k))
               number = number + 1;
            end
        end
        probs(k) = number/(size(img,1));
    end
    [dict, ~]= huffmandict(q_level,probs);
    code = huffmanenco(img,dict);
end
%% Decoding 
function decode = decoding(code,dict,array_width)
    decode = huffmandeco(code,dict);
    decode = reshape(decode,[],array_width);
end