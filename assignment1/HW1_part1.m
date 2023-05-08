% 5LSE0 pratical assignment part 1
% Author: Shao Hsuan Hung (s.hung@student.tue.nl)
% ID: 1723219
% Update date: 1/7/2023
% MATLAB version: R2022a
% Required file: lena.pgm, peppers.pgm, cameraman.pgm, butterfly.pgm
clc;clear;close all;
%% 1.a Show PCM encoding
lena = imread("lena.pgm");
[bbp1,~,~ ] = PCM(lena,1);
[bbp2,~, ~] = PCM(lena,2);
[bbp3,~, ~] = PCM(lena,3);
[bbp4,~,~ ] = PCM(lena,4);
[bbp5,~,~ ] = PCM(lena,5);
[bbp6,~,~ ]= PCM(lena,6);
[bbp7,~, ~]= PCM(lena,7);
[bbp8,~, ~] = PCM(lena,8);
%% 1.b 
lena = double((imread("lena.pgm")))/255;
peppers = double(imread("peppers.pgm"))/255;
bbp_size = 8;
quantize_lena = zeros(512,512,bbp_size);
snr_array_lena = zeros(bbp_size,1);
snr_array_pepper = zeros(bbp_size,1);
for bpp = 1:1:bbp_size
    %
    [A, quantize_lena(:,:,bpp),maxValue, minValue] = uniform_quantizer(lena,bpp);
    %     quantize_lena(:,:,bpp) = test_q(lena,bpp);
    snr_array_lena(bpp)= snr(double(lena),double(quantize_lena(:,:,bpp)-double(lena)));
%     snr_array_lena(bpp) = calculate_snr(double(lena),double(quantize_lena(:,:,bpp)));
    %
    [A, quantize_pepper(:,:,bpp),maxValue, minValue] = uniform_quantizer(peppers,bpp);
    %     quantize_pepper(:,:,bpp) = test_q(peppers,bpp);
    snr_array_pepper(bpp) = snr(double(peppers),double(quantize_pepper(:,:,bpp)-double(peppers)));
%     snr_array_pepper(bpp) = calculate_snr(double(peppers),double(quantize_pepper(:,:,bpp)));
end
set(0,'defaultfigurecolor','w') 
bbp1= cast(quantize_lena(:,:,1),'double');
bbp2= cast(quantize_lena(:,:,2),'double');
bbp3= cast(quantize_lena(:,:,3),'double');
bbp4= cast(quantize_lena(:,:,4),'double');
bbp5= cast(quantize_lena(:,:,5),'double');
bbp6= cast(quantize_lena(:,:,6),'double');
bbp7= cast(quantize_lena(:,:,7),'double');
bbp8= cast(quantize_lena(:,:,8),'double');

figure()
imshow(lena,[min(lena(:)) max(lena(:))]);
figure()
imshow(bbp1,[min(bbp1(:)) max(bbp1(:))]);
figure()
imshow(bbp2,[min(bbp2(:)) max(bbp2(:))]);
figure()
imshow(bbp3,[min(bbp3(:)) max(bbp3(:))]);
figure()
imshow(bbp4,[min(bbp4(:)) max(bbp4(:))]);
figure()
imshow(bbp5,[min(bbp5(:)) max(bbp5(:))]);
figure()
imshow(bbp6,[min(bbp6(:)) max(bbp6(:))]);
figure()
imshow(bbp7,[min(bbp7(:)) max(bbp7(:))]);
figure()
imshow(bbp8,[min(bbp8(:)) max(bbp8(:))]);
fig = figure()
plot(snr_array_lena,"--o","DisplayName","lena.pgm")
title("SNR versus the bitrate after compression")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
hold on;grid on 
plot(snr_array_pepper,"--*","DisplayName","peppers.pgm")
legend
% saveas(fig,"SNR versus the bitrate after compression.eps")
% 
% Normalized & Save the image 
% bbp1 = normalized(bbp1);
% bbp2 = normalized(bbp2);
% bbp3 = normalized(bbp3);
% bbp4 = normalized(bbp4);
% bbp5 = normalized(bbp5);
% bbp6 = normalized(bbp6);
% bbp7 = normalized(bbp7);
% bbp8 = normalized(bbp8);
% imwrite(lena,'lena.png');
% imwrite(bbp1,'bbp1.png');
% imwrite(bbp2,'bbp2.png');
% imwrite(bbp3,'bbp3.png');
% imwrite(bbp4,'bbp4.png');
% imwrite(bbp5,'bbp5.png');
% imwrite(bbp6,'bbp6.png');
% imwrite(bbp7,'bbp7.png');
% imwrite(bbp8,'bbp8.png');
%% 1-c
close all;
lena =double((imread('lena.pgm')))/255;
peppers = double(imread('peppers.pgm'))/255;
lena_uniform_noise = lena + (0.2-0)*rand(size(lena)); % add small uniform noise
peppers_uniform_noise= peppers + (0.2-0)*rand(size(peppers));
lena_noise = lena_uniform_noise - lena;
peppers_noise = peppers_uniform_noise - peppers;

[bbp1_c, ~, ~] = PCM(lena_uniform_noise,1);
[bbp2_c, ~, ~] = PCM(lena_uniform_noise,2);
[bbp3_c, ~, ~] = PCM(lena_uniform_noise,3);
[bbp4_c, ~, ~] = PCM(lena_uniform_noise,4);
[bbp5_c, ~, ~] = PCM(lena_uniform_noise,5);
[bbp6_c, ~, ~] = PCM(lena_uniform_noise,6);
[bbp7_c, ~, ~] = PCM(lena_uniform_noise,7);
[bbp8_c, ~, ~] = PCM(lena_uniform_noise,8);
%
[bbp1_p, decoding1, coding1] = PCM(peppers_uniform_noise,1);
[bbp2_p, decoding2, coding2] = PCM(peppers_uniform_noise,2);
[bbp3_p, decoding3, coding3] = PCM(peppers_uniform_noise,3);
[bbp4_p, decoding4, coding4] = PCM(peppers_uniform_noise,4);
[bbp5_p, decoding5, coding5] = PCM(peppers_uniform_noise,5);
[bbp6_p, decoding6, coding6] = PCM(peppers_uniform_noise,6);
[bbp7_p, decoding7, coding7] = PCM(peppers_uniform_noise,7);
[bbp8_p, decoding8, coding8] = PCM(peppers_uniform_noise,8);
% After uncompress, minus the noise
pictures(:,:,1)= bbp1_c-lena_noise; 
pictures(:,:,2)= bbp2_c-lena_noise;
pictures(:,:,3)= bbp3_c-lena_noise;
pictures(:,:,4)= bbp4_c-lena_noise;
pictures(:,:,5)= bbp5_c-lena_noise;
pictures(:,:,6)= bbp6_c-lena_noise;
pictures(:,:,7)= bbp7_c-lena_noise;
pictures(:,:,8)= bbp8_c-lena_noise;
%
pictures_p(:,:,1)= bbp1_p-peppers_noise;
pictures_p(:,:,2)= bbp2_p-peppers_noise;
pictures_p(:,:,3)= bbp3_p-peppers_noise;
pictures_p(:,:,4)= bbp4_p-peppers_noise;
pictures_p(:,:,5)= bbp5_p-peppers_noise;
pictures_p(:,:,6)= bbp6_p-peppers_noise;
pictures_p(:,:,7)= bbp7_p-peppers_noise;
pictures_p(:,:,8)= bbp8_p-peppers_noise;

snr_array_lena_dither = zeros(8,1);
snr_array_peper_dither = zeros(8,1);

for bpp = 1:1:8
    % SNR original image or dither image 
    snr_array_lena_dither(bpp) = calculate_snr(double(lena),double(pictures(:,:,bpp)));
    snr_array_peper_dither(bpp) = calculate_snr(double(peppers),double(pictures_p(:,:,bpp)));
end 
%% Plot the result of 1-c
% Decommend to show the bunch of the image 
% figure()
% imshow(lena,[min(lena(:)) max(lena(:))]);
% fgc1 = figure();
% imshow(pictures(:,:,1),[min(min(pictures(:,:,1))) max(max(pictures(:,:,1)))]);
% fgc2 = figure();
% imshow(pictures(:,:,2),[min(min(pictures(:,:,2))) max(max(pictures(:,:,2)))]);
% fgc3 = figure();
% imshow(pictures(:,:,3),[min(min(pictures(:,:,3))) max(max(pictures(:,:,3)))]);
% fgc4 = figure();
% imshow(pictures(:,:,4),[min(min(pictures(:,:,4))) max(max(pictures(:,:,4)))]);
% fgc5 = figure();
% imshow(pictures(:,:,5),[min(min(pictures(:,:,5))) max(max(pictures(:,:,5)))]);
% fgc6 = figure();
% imshow(pictures(:,:,6),[min(min(pictures(:,:,6))) max(max(pictures(:,:,6)))]);
% fgc7 = figure();
% imshow(pictures(:,:,7),[min(min(pictures(:,:,7))) max(max(pictures(:,:,7)))]);
% fgc8 = figure();
% imshow(pictures(:,:,8),[min(min(pictures(:,:,8))) max(max(pictures(:,:,8)))]);

% lena
dither_1 = normalized(pictures(:,:,1));
dither_2 = normalized(pictures(:,:,2));
dither_3 = normalized(pictures(:,:,3));
dither_4 = normalized(pictures(:,:,4));
dither_5 = normalized(pictures(:,:,5));
dither_6 = normalized(pictures(:,:,6));
dither_7 = normalized(pictures(:,:,7));
dither_8 = normalized(pictures(:,:,8));
dither_lena_array(:,:,1) = dither_1;
dither_lena_array(:,:,2) = dither_2;
dither_lena_array(:,:,3) = dither_3;
dither_lena_array(:,:,4) = dither_4;
dither_lena_array(:,:,5) = dither_5;
dither_lena_array(:,:,6) = dither_6;
dither_lena_array(:,:,7) = dither_7;
dither_lena_array(:,:,8) = dither_8;
% Save the image 
% imwrite(dither_1,'bbp1_dither.png');
% imwrite(dither_2,'bbp2_dither.png');
% imwrite(dither_3,'bbp3_dither.png');
% imwrite(dither_4,'bbp4_dither.png');
% imwrite(dither_5,'bbp5_dither.png');
% imwrite(dither_6,'bbp6_dither.png');
% imwrite(dither_7,'bbp7_dither.png');
% imwrite(dither_8,'bbp8_dither.png');
% peppers
dither_1_p = normalized(pictures_p(:,:,1));
dither_2_p = normalized(pictures_p(:,:,2));
dither_3_p = normalized(pictures_p(:,:,3));
dither_4_p = normalized(pictures_p(:,:,4));
dither_5_p = normalized(pictures_p(:,:,5));
dither_6_p = normalized(pictures_p(:,:,6));
dither_7_p = normalized(pictures_p(:,:,7));
dither_8_p = normalized(pictures_p(:,:,8));
dither_pepper_array(:,:,1) = dither_1_p;
dither_pepper_array(:,:,2) = dither_2_p;
dither_pepper_array(:,:,3) = dither_3_p;
dither_pepper_array(:,:,4) = dither_4_p;
dither_pepper_array(:,:,5) = dither_5_p;
dither_pepper_array(:,:,6) = dither_6_p;
dither_pepper_array(:,:,7) = dither_7_p;
dither_pepper_array(:,:,8) = dither_8_p;
figure
plot(snr_array_lena,"--o","DisplayName","lena.pgm")
hold on; grid on;
plot(snr_array_lena_dither,"--*","DisplayName","lena.pgm (Dither PCM compressed image)")
title("SNR versus the bitrate after PCM on lena.pgm")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend

figure
plot(snr_array_pepper,"--o","DisplayName","peppers.pgm")
hold on; grid on;
plot(snr_array_peper_dither,"--*","DisplayName","pepper.pgm (Dither PCM compressed image)")
title("SNR versus the bitrate after PCM, on pepper.pgm")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend
%-------------------  Below here, are PCM function -----------------%
%% Implement PCM
function [q_img,decode,coding] = PCM(img,bpp)
% Input:
% img : uint8 image
% bpp : bit pre pixel
% Return:
% q_img: image after compression
% decode: decode of the image 
% coding:Encoded of the image
    % 1. Quantizer: uniform quantizer
    [q_level, q_img] = uniform_quantizer(img,bpp);
    %%%%%%% show the quantized image %%%%%%%%%%%%%%%
%       figure()
%       imshow(q_img,[min(q_img(:)) max(q_img(:))]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2. Encoding
    [dict,coding] = encoding(q_img,q_level,bpp);

    % 3. Decoding
     decode = decoding(coding,dict);
    % 4. Dequantized the image would return the same q_imag
end

%% -------------- Below here, are subfunction used in the PCM --------------%
% Quantizer
function [q_level, new_img, maxValue, minValue] = uniform_quantizer(img, bit_rate)
    % Normalized the image 
%     img = normalized(double(img));
    maxValue =  max(max(img));
    minValue = min(min(img));
    delta = (maxValue - minValue)/(2^bit_rate);
    q_level = zeros(bit_rate,1);
    for i =1:1:2^bit_rate
        q_level(i) = minValue + delta*(i);
    end
    new_img = zeros(size(img));
    for j = 1:1:size(img,1)
         for k = 1:1:size(img,2)
               for i = 1:1:2^bit_rate
                    if(img(j,k) <= q_level(i))
                        new_img(j,k) = q_level(i);
                        break
                    end
                    if(img(j,k) >= q_level(2^bit_rate))
                        new_img(j,k) = q_level(2^bit_rate);
                    end
               end
         end
    end
end
%% Encoding 
function [dict,code] = encoding(img,q_level,bit_rate)
    img = reshape(img,[],1);
    q_level = reshape(q_level,[],1);
    probs = zeros(2^bit_rate,1);
    for k = 1:1:size(q_level,1)
        number = 0;
        for i = 1:1:size(img,1)
            if(img(i)==q_level(k))
               number = number + 1;
            end
        end
        probs(k) = number/(size(img,1));
    end
    [dict, ~]= huffmandict(q_level,probs);
    code = huffmanenco(img,dict);
end
%% Implement the SNR
function snr = calculate_snr(ori_img, compress_img)
    mse = 0;
    square = 0;
    for i = 1:1:size(compress_img,1)
        for j = 1:1:size(compress_img,2)
                square = square + (compress_img(i,j))^2;
                mse = mse + (ori_img(i,j)-compress_img(i,j))^2;
        end
    end
    snr = 10*log10(square/mse);
%     snr = 10*log10((std(ori_img,1,'all'))^2/(std(error,1,'all')^2));
end
%% Decoding 
function decode = decoding(code,dict)
    decode = huffmandeco(code,dict);
end
%% Normalizaed image
function new_img = normalized(img)
    img = double(img);
    new_img = (img-min(img(:)))/(max(img(:))-min(img(:)));
end