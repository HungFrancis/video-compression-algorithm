% 5LSE0 pratical assignment part 1
% Author: Shao Hsuan Hung (s.hung@student.tue.nl)
% ID: 1723219
% Update date: 1/7/2023
% MATLAB version: R2022a
% Required file: lena.pgm, peppers.pgm, cameraman.pgm, butterfly.pgm
clc;clear;close all
lena = double(imread("lena.pgm"));
peppers = double(imread("peppers.pgm"));
%% Pipeline II
bit_rate = 8;
snr_array_lena = zeros(bit_rate,1);
snr_array_pepper = zeros(bit_rate,1);
error_map(:,:,:) = zeros(size(lena,1),size(lena,2),bit_rate);
error_map_pepper(:,:,:) = zeros(size(lena,1),size(lena,2),bit_rate);
compress_img(:,:,:) = zeros(size(lena,1),size(lena,2),bit_rate);
compress_img_pepper(:,:,:) = zeros(size(lena,1),size(lena,2),bit_rate);
for bpp = 1:1:bit_rate
    %
    tic
    [error_encode, error_dict, error_map(:,:,bpp)] = DPCM_encoder(lena,bpp);
    [error_encode_pepper, error_dict_pepper, error_map_pepper(:,:,bpp)] = DPCM_encoder(peppers,bpp);
    % Show the error image
%     figure
%     imshow(error_map(:,:,bpp),[])

    % Decode
    compress_img(:,:,bpp) = DPCM_decoder(error_encode,error_dict);
    compress_img_pepper(:,:,bpp) = DPCM_decoder(error_encode_pepper,error_dict_pepper);
    % Calculate the SNR
    snr_array_lena(bpp) = calculate_snr(lena, compress_img(:,:,bpp));
    snr_array_pepper(bpp) = calculate_snr(peppers,compress_img_pepper(:,:,bpp));
    % Normalize the image
    compress_img(:,:,bpp) = normalized(compress_img(:,:,bpp));
    toc
    % Show the compress images
%     figure
%     imshow(compress_img(:,:,bpp),[])
end
% Store the image 
imwrite(compress_img(:,:,1),'bbp1_DPCM.png');
imwrite(compress_img(:,:,2),'bbp2_DPCM.png');
imwrite(compress_img(:,:,3),'bbp3_DPCM.png');
imwrite(compress_img(:,:,4),'bbp4_DPCM.png');
imwrite(compress_img(:,:,5),'bbp5_DPCM.png');
imwrite(compress_img(:,:,6),'bbp6_DPCM.png');
imwrite(compress_img(:,:,7),'bbp7_DPCM.png');
imwrite(compress_img(:,:,8),'bbp8_DPCM.png');
%
figure
plot(snr_array_lena,"--o","DisplayName","lena.pgm");
hold on; grid on;
plot(snr_array_pepper,"--*","DisplayName","peppers.pgm");
title("SNR versus the bitrate after DPCM compression (Pipeline (II))")
xlabel("bits per pixel (bpp)")
ylabel("SNR (dB)")
legend
%% 2-b subjective picture quality
DCT_quality = [2 2 3 3 3 4 4 4];
DPCM_quality = [3 3 3 4 4 5 5 5];
figure
plot(DCT_quality,'--o',"DisplayName","DCT, pipeline(I)");
hold on; grid on;
plot(DPCM_quality,'--*',"DisplayName","DPCM, pipeline(II)");
title("Subjective picture quality w.r.t the change of the bit rate")
xlabel("bits per pixel (bpp)");
ylabel("Subjective picture quality score");
legend
ylim([1 5])
set(gca, 'YTick', 1:5);
%% 2-c computation time
DCT_computation_time = [0.09 0.076 0.076 0.072 0.081 0.086 0.14 0.12 0.15 0.24 0.40 0.74 1.51 2.7 5.18 10.55]
DPCM_computation_time  = [1.0726 1.246496 1.36 1.44 1.66 2.34 6.2686 4.564 7.4128 12.1213 23.32 65.76 180.34 689.375]
figure
semilogy(DCT_computation_time,'--o','DisplayName','DCT pipeline(I)');
hold on; grid on;
semilogy(DPCM_computation_time,'--*','DisplayName','DPCM pipeline(II)');
legend
xlabel('bits per pixel (bpp)');ylabel('logarithm Computation time (s)');
title('Comparison of computation time between pipeline(I) and pipeline(II)')
%%
function [quantized_error_code, decode_dict, quantized_error] = DPCM_encoder(img,bpp)
    % 1. Computer the error
    % 2. Quantized the error
    % 3. Add the quantized error to the predict and keep this value to be used
    % by the predictor or the following samples
    % A, B, C is the parameters to be tune
    A = 0.75;B=0.75;C=-(0.75^2);
    predictor = zeros(size(img,1),1);
    quantized_error = zeros(size(img,1),1);
    max_img = max(img(:));
    min_img = min(img(:));
    % Predictor use the adjacent error elements in prvious row, col
    for i = 1:1:size(img,1)
        % Read row 
        for j = i:size(img,2)
            if i==1 % First row 
                if j ==1 % First column
                    predicted = 0; % the predicted value, but img(1,1) is the first element, not predicted value
                else
                    predicted = A*predictor(i,j-1);
                end
            else
                predicted = A*predictor(i-1,j)+B*predictor(i,j-1)+C*predictor(i-1,j-1);
            end
            error = img(i,j) - predicted;
            % Quantize the error
            quantized_error(i,j)  = quantize_error(error,bpp,max_img,min_img);
            predictor(i,j) = predicted + quantized_error(i,j);
        end
        % Read Column
        for j = (i+1): size(img,2)
            if i == 1
                predicted = A*predictor(j-1,i);
            else
                predicted = A*predictor(j-1,i)+B*predictor(j,i-1)+C*predictor(j-1,i-1);
            end
            error = img(j,i) - predicted;
            % Quantize the error
            quantized_error(j,i) = quantize_error(error,bpp,max_img,min_img);
            predictor(j,i) = predicted + quantized_error(j,i);
        end
    end
    % Then encode the errormap
    Min = -255; Max = 255; delta = (Max-Min)/(2^bpp);
    q_level = zeros(2^bpp,1);
    for i = 1:1:2^bpp
        q_level(i) = Min + delta*0.5 + delta*(i-1);
    end
    [decode_dict,quantized_error_code] = encoding(quantized_error,q_level,bpp);
end
%%
function [quantized_error] = quantize_error(error, bpp,max_img,min_img)
% the input error is just a value
% The range of the image is 0~255, so:
% The range of error would be -255 ~ 255    
    min = -255;
    max = 255;
    delta = (max-min)/(2^bpp);
    error_level = min;
    q_level = zeros(2^bpp,1);
    for i = 1:1:(2^bpp)
         q_level(i) = min + delta*0.5 + delta*(i-1);
         q_max = max - delta*0.5;
         if( error <= q_level(i) )
             quantized_error =  q_level(i);
             break;
         elseif( error >= q_level(2^bpp) )
             quantized_error =  q_level(2^bpp) ;
         end
    end
end
%% Decoder
function [compress_img] = DPCM_decoder(error_encode,decode_dict)
    % Step1. Decode the error map  
    decode_error = decoding(error_encode, decode_dict);
    % Step 2. Follow the DPCM decoder scheme, compensate the error
    % Read the image row1, column1, row2, column2, row3, ... because the
    % predictor uses the adjacent elements in the previous row and column
    N = size(decode_error,1);
    compress_img = zeros(N);
    predictor = zeros(N);
    % Reproduce the predictor
    A = 0.75;B=0.75;C=-(0.75^2); % Parameters to adjust
    for i=1:N
        % Read row i
        for j=i:N
            if i==1
                if j==1
                    predicted = 0;
                else
                    predicted = A*predictor(i,j-1);
                end
            else
                predicted = A*predictor(i-1,j) +  B*predictor(i,j-1)+C*predictor(i-1,j-1);
            end 
            compress_img(i,j) = predicted + decode_error(i,j);
            predictor(i,j) = predicted + decode_error(i,j);
        end
        
        % Read column i
        for j=(i+1):N
            if i==1
                predicted = A*predictor(j-1,i);
            else
                predicted =A*predictor(j-1,i) +  B*predictor(j,i-1) + C*predictor(j-1,i-1);
            end
            
            compress_img(j,i) = predicted + decode_error(j,i);
            predictor(j,i) = predicted + decode_error(j,i);
        end
    end
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
%%
function new_img = normalized(ori_img)
    img = double(ori_img);
    new_img = (img-min(img(:)))/(max(img(:))-min(img(:)));
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
%% Decoding 
function decode = decoding(code,dict)
    decode = huffmandeco(code,dict);
    decode = reshape(decode,[],512);
end