% 5LSE0 pratical assignment part 1
% Author: Shao Hsuan Hung (s.hung@student.tue.nl)
% ID: 1723219
% Update date: 1/7/2023
% MATLAB version: R2022a
% Required file: lena.pgm, peppers.pgm, cameraman.pgm, butterfly.pgm
clc;clear;close all
lena = double(imread("lena.pgm"))/255;
peppers = double(imread("peppers.pgm"))/255;
%% Test the pipeline (I)
bit_rate = 8;
snr_array_lena= zeros(bit_rate,1);
snr_array_pepper= zeros(bit_rate,1);
idct = zeros(512,512,bit_rate);
idct_peppers = zeros(512,512,bit_rate);
for bpp = 1:1:bit_rate
    tic
    [snr_value , idct_img] = DCT_pipeline1(lena,bpp);
    toc
    [snr_value_pepper, idct_img_pepper] = DCT_pipeline1(peppers,bpp);
    idct(:,:,bpp) = normalized(idct_img); % Normalize the idct before save the image
    idct_peppers(:,:,bpp) = idct_img_pepper;
    snr_array_lena(bpp) = snr_value;
    snr_array_pepper(bpp) = snr_value_pepper;
end

imwrite(idct(:,:,1),'bbp1_DCT.png');
imwrite(idct(:,:,2),'bbp2_DCT.png');
imwrite(idct(:,:,3),'bbp3_DCT.png');
imwrite(idct(:,:,4),'bbp4_DCT.png');
imwrite(idct(:,:,5),'bbp5_DCT.png');
imwrite(idct(:,:,6),'bbp6_DCT.png');
imwrite(idct(:,:,7),'bbp7_DCT.png');
imwrite(idct(:,:,8),'bbp8_DCT.png');

figure
plot(snr_array_lena,"--o","DisplayName","lena.pgm");
hold on; grid on;
plot(snr_array_pepper,"--*","DisplayName","peppers.pgm");
title("SNR versus the bitrate after DCT compression (Pipeline (I))")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend
% imshow(idct(:,:,8),[])
%% Pipeline 1, 
% source -> DCT -> Quantize coeff. -> Dequantize coeff. -> IDCT
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
    % Step5. rescale the idct image 
    snr_value = calculate_snr(img,idct_img);
end
% -------------------- Below are subfuntion ----------------%
%% Quantizer
function [q_level, new_img, delta] = DCT_quantize(img,bit_rate)
    maxValue =  max(max(img));
    minValue = min(min(img));
    delta = (maxValue - minValue)/(2^bit_rate);
    q_level = zeros(bit_rate,1);
    for i =1:1:2^bit_rate
        q_level(i) =  minValue + delta*(i-1);
    end
    new_img = zeros(size(img));
    new_img = (fix(img./delta));
end
%% Dequantizer
function de_img = DCT_dequantize(img,delta)
    de_img = img*delta+delta/2;
end
%% Implement the SNR
function snr = calculate_snr(ori_img, compress_img)
    mse = 0;
    square = 0;
    ori_img = normalized_img(ori_img);
    compress_img = normalized_img(compress_img);
    for i = 1:1:size(compress_img,1)
        for j = 1:1:size(compress_img,2)
                square = square + (compress_img(i,j))^2;
                mse = mse + (ori_img(i,j)-compress_img(i,j))^2;
        end
    end
    snr = 10*log10(square/mse);
end
%% 
function new_img = normalized(img)
    new_img = (img-min(img(:)))./(max(img(:)-min(img(:))));
end

%%
function [new_img , mean_img, std_img] = normalized_img(ori_img)
    img = double(ori_img);
    mean_img = mean2(img);
    std_img = std2(ori_img);
    new_img = (ori_img - mean_img)./std_img;
end
%% 
function new_img = denormalized_img(ori_img,mean_img,std_img)
    new_img = (ori_img.*std_img)+mean_img;
end