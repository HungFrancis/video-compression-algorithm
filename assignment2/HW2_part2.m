clc;clear;close all;
lena = im2double(imread('lena.pgm'));
pepper = im2double(imread('peppers.pgm'));
%% 2.a
LENA = fftn(lena);
LENA = log(1+fftshift(LENA)); % Resclaing
PEPPER = fftn(pepper);
PEPPER = log(1+fftshift(PEPPER));
%
amp_lena = abs(LENA);
phase_lena = angle(LENA);
amp_pepper = abs(PEPPER);
phase_pepper = angle(PEPPER);
%
figure
subplot(1,3,1)
imshow(lena,[])
title('Oringinal Image')
subplot(1,3,2)
imshow(amp_lena,[])
title('Amplitude Specturm')
subplot(1,3,3)
imshow(phase_lena,[])
title('Phase Specturm')

figure
subplot(1,3,1)
imshow(pepper,[])
title('Oringinal Image')
subplot(1,3,2)
imshow(amp_pepper,[])
title('Amplitude Specturm')
subplot(1,3,3)
imshow(phase_pepper,[])
title('Phase Specturm')
%% 2.b
LENA = fft2((lena));
PEPPER = fft2((pepper));
amp_lena = abs(LENA);
phase_lena = angle(LENA);
amp_pepper = abs(PEPPER);
phase_pepper = angle(PEPPER);
% AMP: lena, Phase: pepper
img_combine1 = amp_lena.*exp(1j*phase_pepper);

img_combine2 = amp_pepper.*exp(1j*phase_lena);
inver_combine1 = abs(ifft2(img_combine1));
inver_combine2 = abs(ifft2(img_combine2));
figure
imshow(inver_combine1,[min(min(inver_combine1)) max(max(inver_combine1))])
colormap gray

figure
imshow(inver_combine2,[min(min(inver_combine1)) max(max(inver_combine1))])
colormap gray

imwrite(inver_combine1,'magLena_phapepper.png')
imwrite(inver_combine2,'magpepper_phaLena.png')
%% 2.c
lena = im2double(imread('lena.pgm'));
pepper = im2double(imread('peppers.pgm'));
patches = zeros(8,8,1000);
number_x = randi([1 512-8],1,1000);
number_y = randi([1 512-8],1,1000);
mean_of_block = zeros(1000,1);
resize_patches = zeros(64,1000);
for i = 1:1:1000
    patches(:,:,i) = lena(number_x(i):number_x(i)+7,number_y(i):number_y(i)+7);
    % DC centering 
    mean_of_block(i) = mean2(patches(:,:,i));
    patches(:,:,i) = patches(:,:,i) - mean_of_block(i);
    % Contrast normalize: x-mu/std
    patches(:,:,i) = (patches(:,:,i))/std2(patches(:,:,i));
    patches(:,:,i) = rescale(patches(:,:,i),0,1);
    % Resize the patches for the 2.d
    %patches(:,:,i) = rescale(patches(:,:,i),0,1);
    resize_patches(:,i) = reshape(patches(:,:,i),64,[]);
end
% for i = 1:10
%     figure
%     imshow(patches(:,:,i),[])
%     imwrite(im2uint8(mat2gray(patches(:,:,i))), gray,strcat(num2str(i),'_2c.jpg'))     
% end
%% 2.d
cov_matrix = cov(resize_patches');
[eig_vector eig_num]= eig(cov_matrix);
imshow(eig_vector,[])
[eig_dig,idx] = sort(diag(eig_num),'descend');
eig_num = diag(eig_dig);
eig_vector = eig_vector(:,idx); 
figure
for i = 1:64
    subplot(8,8,i);
    % Print the component
    imshow(reshape(eig_vector(:,i), 8,8), []);
end
%% 2.e
% In the previous question, we have found the approximatly basis function
% of iamge, now devide the imaage in to 8*8*4096 block and then whitening
% them with the eigen vector and eigen value
% Prepare the 8*8*4096 from the lena image 
block_lena = zeros(8,8,size(lena,1)*size(lena,2)/64);
whiten_lena = zeros(size(lena));
for i = 1:1:512/8
    for j = 1:1:512/8
        block_lena(:,:,64*(i-1)+j) = lena(8*(j-1)+1:8*(j-1)+8,8*(i-1)+1:8*(i-1)+8);
        % Contrast Normalization
        block_lena(:,:,64*(i-1)+j) = (block_lena(:,:,64*(i-1)+j)-mean2(block_lena(:,:,64*(i-1)+j)))/std2(block_lena(:,:,64*(i-1)+j));
    end
end
flatten_block_lena = reshape(block_lena,64,[]);
% Whiten the patches
diagonal_eigen_matrix = diag(diag(eig_num).^(-0.5));
W_whiten = eig_vector*diagonal_eigen_matrix*eig_vector';
whiten_patches = W_whiten*flatten_block_lena;
% Recover the image 
whiten_block_lena = reshape(whiten_patches,8,8,[]);
for i = 1:1:512/8
    for j = 1:1:512/8
        whiten_lena(8*(j-1)+1:8*(j-1)+8,8*(i-1)+1:8*(i-1)+8) = whiten_block_lena(:,:,64*(i-1)+j);
    end
end
figure
imshow(whiten_lena,[])
imwrite(whiten_lena,'whiten_lena.jpg')
whiteh_cov = cov(whiten_lena);
figure
imshow(whiteh_cov,[])