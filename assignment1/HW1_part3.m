% 5LSE0 pratical assignment part 1
% Author: Shao Hsuan Hung (s.hung@student.tue.nl)
% ID: 1723219
% Update date: 1/7/2023
% MATLAB version: R2022a
% Required file: lena.pgm, peppers.pgm, cameraman.pgm, butterfly.pgm
%% Part3, DCT, frequency domain
clc;clear;close all
lena = double(imread("lena.pgm"));
peppers = double(imread("peppers.pgm"));
%
step_size = 8;
brightness_dct_coeff_img(:,:,:) = zeros(size(lena,1),size(lena,2),step_size);
contrast_dct_coeff_img(:,:,:) = zeros(size(lena,1),size(lena,2),step_size);
idct_img_brigthness(:,:,:) = zeros(size(lena,1),size(lena,2),step_size);
idct_img_contrast(:,:,:) = zeros(size(lena,1),size(lena,2),step_size);
idct_img_rotate(:,:,:) = zeros(size(lena,1),size(lena,2),step_size);
%
dct_block = dctmtx(8);
idct = @(block_struct) dct_block'*(block_struct.data)*dct_block;
dct = @(block_struct) dct_block*(block_struct.data)*dct_block';
%
dct_coeff_img = (blockproc(lena,[8 8],dct));
%% 3.a Change of picture brightness with loop
% First -90
delta_brightness1 = ones(512)*(-90);
DCT_delta_brightness1 = blockproc(delta_brightness1,[8 8],dct);
brightness_dct_coeff_img1 = dct_coeff_img + DCT_delta_brightness1;
idct_img_brigthness1 =blockproc(brightness_dct_coeff_img1,[8 8],idct);
idct_img_brigthness1  = cast(idct_img_brigthness1,'uint8');
% second +90
delta_brightness2 = ones(512)*(90);
DCT_delta_brightness2 = blockproc(delta_brightness2,[8 8],dct);
brightness_dct_coeff_img2 = dct_coeff_img + DCT_delta_brightness2;
idct_img_brigthness2 =blockproc(brightness_dct_coeff_img2,[8 8],idct);
idct_img_brigthness2  = cast(idct_img_brigthness2,'uint8');
f = figure;
subplot(1,3,1)
imshow(idct_img_brigthness1)
title("Brightness -90")
subplot(1,3,2)
imshow(lena,[])
title("Original Image")
subplot(1,3,3)
imshow(idct_img_brigthness2)
title("Brightness +90")
%% 3.b Contrast
% close all; clear;
lena = double(imread("lena.pgm"));
lena = cast(lena,'uint8');
% DCT
dct_coeff_img = dct2(lena);
% Modify DCT coeff. 
contrast_dct_coeff_img1 = dct_coeff_img*(0.5);
idct_img_contrast1 = idct2(contrast_dct_coeff_img1);
idct_img_contrast1 = cast(idct_img_contrast1,'uint8');
%
contrast_dct_coeff_img2 = dct_coeff_img*(1.5);
idct_img_contrast2 = idct2(contrast_dct_coeff_img2);
idct_img_contrast2 = cast(idct_img_contrast2,'uint8');
figure
subplot(1,3,1)
imshow(idct_img_contrast1)
title("Contrast c<1 (c = 0.5)")
subplot(1,3,2)
imshow(lena,[])
title("Original Image")
subplot(1,3,3)
imshow(idct_img_contrast2)
title("Contrast c>1 (c = 1.5)");
%% 3.c Rotate by 180 degree
lena = double(imread("lena.pgm"));
H = [1 -1 1 -1 1 -1 1 -1;
     -1 1 -1 1 -1 1 -1 1;
     1 -1 1 -1 1 -1 1 -1;
     -1 1 -1 1 -1 1 -1 1;
     1 -1 1 -1 1 -1 1 -1;
     -1 1 -1 1 -1 1 -1 1;
     1 -1 1 -1 1 -1 1 -1;
     -1 1 -1 1 -1 1 -1 1;]; %if u+v = odd, *(-1)
% function handler
dct = @(block_struct) dct2(block_struct.data);
idct = @(block_struct) idct2(block_struct.data);
odd_pos_minus_one = @(block_struct) (block_struct.data).*H;
central_sym = @(block_struch) block_struch.data(end:-1:1,end:-1:1)';
% DCT
dct_coeff_img = (blockproc(lena,[8 8],dct));
% Rotate
for i = 1:512/8
    for j = 1:512/8
        dct_coeff_img_revese(1+8*(i-1):1+8*(i-1)+7,1+8*(j-1):1+8*(j-1)+7) = dct_coeff_img(512-8*i+1:512-8*i+8,512-8*j+1:512-8*j+8);
    end
end
rotate_dct_coeff_img = blockproc(dct_coeff_img_revese,[8 8],odd_pos_minus_one);
% IDCT
rotate_img = blockproc(rotate_dct_coeff_img,[8 8],idct);
figure
subplot(1,2,1)
imshow(lena,[])
title("Original image")
subplot(1,2,2)
imshow(cast(rotate_img,'uint8'),[])
title("Rotate the image by 180 degree")
%%
% IDCT with the new coefficient
% Result
imwrite(idct_img_brigthness(:,:,1),'brightness1.png');
imwrite(idct_img_brigthness(:,:,2),'brightness2.png');
imwrite(idct_img_brigthness(:,:,3),'brightness3.png');
imwrite(idct_img_brigthness(:,:,4),'brightness4.png');
imwrite(idct_img_brigthness(:,:,5),'brightness5.png');
imwrite(idct_img_brigthness(:,:,6),'brightness6.png');
imwrite(idct_img_brigthness(:,:,7),'brightness7.png');
imwrite(idct_img_brigthness(:,:,8),'brightness8.png');
% Change of image contrast
imwrite(idct_img_contrast(:,:,1),'Contrast1.png');
imwrite(idct_img_contrast(:,:,2),'Contrast2.png');
imwrite(idct_img_contrast(:,:,3),'Contrast3.png');
imwrite(idct_img_contrast(:,:,4),'Contrast4.png');
imwrite(idct_img_contrast(:,:,5),'Contrast5.png');
imwrite(idct_img_contrast(:,:,6),'Contrast6.png');
imwrite(idct_img_contrast(:,:,7),'Contrast7.png');
imwrite(idct_img_contrast(:,:,8),'Contrast8.png');
%%
function new_img = normalized(ori_img)
    img = double(ori_img);
    new_img = (img-min(img(:)))/(max(img(:))-min(img(:)));
end