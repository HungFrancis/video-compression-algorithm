%% 1.a
clc;clear;close all
set(0,'defaultfigurecolor','w') 
[X,Y]=meshgrid(-5:0.1:5);
subplot(1,2,1)
imagesc(X)
set(gca, 'YDir', 'normal');
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(X, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(X, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
title('Matrice X');colormap gray
subplot(1,2,2)
imagesc(Y)
set(gca, 'YDir', 'normal');
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(Y, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(Y, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
title('Matrice Y')
colormap gray
%% 1.b
[X,Y]=meshgrid(-5:0.01:5);
[cosine,sine]= gabor_spatial(X,Y,-pi/4,1,0.5,0.5);
[U,V]=meshgrid(-0.8:0.01:0.8); % f0 is 0.5
map = gabor_freq_domain(U,V,-pi/4,1,0.5,0.5);
figure
subplot(1,2,1)
imagesc(cosine)
set(gca, 'YDir', 'normal');
colormap(gray)
title("Real part (Spatial Domain)");
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(cosine, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(cosine, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
subplot(1,2,2)
imagesc(sine)
set(gca, 'YDir', 'normal');
colormap(gray)
title("Imginary part (Spatial Domain)");
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(sine, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(sine, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
% subplot(1,3,3)
% imagesc(map)
% title("Frequency Domain")
% set(gca, 'YDir', 'normal');
% xlabel('Horizontal Frequency [rad/pixel]'); ylabel('Vertical Frequency [rad/pixel]');
% yticks = linspace(1, size(map, 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% xticklabels = -0.8:0.2:0.8;
% xticks = linspace(1, size(map, 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)))
colormap("gray")
%% 1.c
clear;
[X,Y]=meshgrid(-5:0.01:5);
[cosine,sine]= gabor_kernel_ChangeXY(X,Y,-pi/4,1,0.5,0.5);
[U,V]=meshgrid(-0.8:0.01:0.8); % f0 is 0.5
map = gabor_freq_domain(U,V,-pi/4,2,1,0.5);
figure
subplot(1,2,1)
imagesc(cosine)
set(gca, 'YDir', 'normal');
colormap(gray)
title("Real part (Spatial Domain)")
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(sine, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(sine, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
subplot(1,2,2)
imagesc(sine)
set(gca, 'YDir', 'normal');
colormap(gray)
title("Imginary part (Spatial Domain)")
yticklabels = 0.5:-0.1:-0.5;
yticks = linspace(1, size(sine, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
xticklabels = 0.5:-0.1:-0.5;
xticks = linspace(1, size(sine, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
% subplot(1,3,3)
% imagesc(map)
% title("Frequency Domain")
% set(gca, 'YDir', 'normal');
% xlabel('Horizontal Frequency [rad/pixel]'); ylabel('Vertical Frequency [rad/pixel]');
% yticklabels = -0.8:0.2:0.8;
% yticks = linspace(1, size(map, 1), numel(yticklabels));
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% xticklabels = -0.8:0.2:0.8;
% xticks = linspace(1, size(map, 2), numel(xticklabels));
% set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)))
% colormap("gray")
%% 1.d 4 oriented in 3 scale
close all
[X,Y]=meshgrid(-5:0.01:5);
[U,V]=meshgrid(-0.8:0.01:0.8); % f0 is 0.1
%
cosine= zeros(size(X,1),size(X,2),4,3);
sine = zeros(size(X,1),size(X,2),4,3);
map = zeros(size(U,1),size(U,2),4,3);
sum_map = zeros(size(U,1),size(U,2));
theta = 0;
for i = 1:1:4
   for j = 1:1:3
       % Parameter setting
       eta = ((2*4)/pi^2)*sqrt(-log(1/sqrt(2)));
       gamma = 3*(log(sqrt(2)))^0.5/pi;
       f0 = 0.5/(2^(j-1));
       % Rotate
       [cosine(:,:,i,j),sine(:,:,i,j)]= gabor_spatial(X,Y,theta,gamma,eta,f0);
       map(:,:,i,j)= gabor_freq_domain(U,V,theta,gamma,eta,f0);
       %
       [c h] = contour(map(:,:,i,j),[max(max(map(:,:,i,j)))/sqrt(2) max(max(map(:,:,i,j)))/sqrt(2)]);
       hold on
       h.LineWidth=1
       h.LineColor = 'b'
       xlabel('Horizontal Frequency [rad/pixel]'); ylabel('Vertical Frequency [rad/pixel]');
       yticklabels = 0.8:-0.2:-0.8;
       yticks = linspace(1, size(map, 1), numel(yticklabels));
       set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
       xticklabels = 0.8:-0.2:-0.8;
       xticks = linspace(1, size(map, 2), numel(xticklabels));
       set(gca, 'XTick', xticks, 'XTickLabel', flipud(xticklabels(:)));
   end
   theta= theta+ (pi/(4));
end
% Uncomment to see the 12 filters 
% for i = 1:1:4
%     for j = 1:1:3
%         figure
%         imagesc(cosine(:,:,i,j));
%         title(3*(i-1)+j)
%         colormap gray
%         imwrite(ind2rgb(im2uint8(mat2gray(cosine(:,:,i,j))), gray),strcat(num2str(3*(i-1)+j),'.jpg'))
%     end
% end
%% 1.b function
function [kernel_real, kernel_img] = gabor_spatial(x,y,theta,gamma,eta,f0)
    x_rotate = x*cos(theta)+y*sin(theta);
    y_rotate = -x*sin(theta)+y*cos(theta);
    g = (f0/(pi*gamma*eta)).*exp(-((f0^2/gamma^2).*x_rotate.^2+(f0^2/eta^2).*y_rotate.^2));
    kernel_real = g.*cos(2*pi*f0.*y_rotate);
    kernel_img = g.*sin(2*pi*f0.*y_rotate);
end

function [kernel_map]= gabor_freq_domain(u,v,theta,gamma,eta,f0)
    u_rotate = u*cos(theta)+v*sin(theta);
    v_rotate = -u*sin(theta)+v*cos(theta);
    kernel_map = exp(-(pi/f0)^2.*(gamma^2.*(u_rotate-f0).^2 + eta^2.*(v_rotate.^2)));
end

function [kernel_real, kernel_img] = gabor_kernel_ChangeXY(x,y,theta,gamma,eta,f0)
    x_rotate = x*cos(theta)+y*sin(theta);
    y_rotate = -x*sin(theta)+y*cos(theta);
    %  Change the roles of X and Y
    tmp = y_rotate;y_rotate = x_rotate;x_rotate = tmp;
    g = (f0/(pi*gamma*eta)).*exp(-((f0^2/gamma^2).*x_rotate.^2+(f0^2/eta^2).*y_rotate.^2));
    kernel_real = g.*cos(2*pi*f0.*y_rotate);
    kernel_img = g.*sin(2*pi*f0.*y_rotate);
end
