%% Load the data set
clc;clear;close all;
test_file = './mnist/scrambled_test.bin';
fid = fopen(test_file, 'r');

[number, count] = fread(fid, 1, 'int32');
if count ~= 1
    disp('failed to read number');
end

[test_permutation, count] = fread(fid, number, 'int32');
if count ~= test_permutation
    disp('failed to read number');
end


[test_labels, count] = fread(fid, number, 'uchar');
if count ~= number
    disp('failed to read number');
end
    
test_digits = fread(fid, [28, 28 * number], 'uchar');
test_digits = reshape(test_digits, [28, 28, number]);

fclose(fid);
disp('loaded test digits');
%% 3.a prepare row base state walking dataset: 28*10000
row_based_walk = zeros(size(test_digits,1),size(test_digits,3));
for i = 1:size(row_based_walk,2)
    for j = 1:size(row_based_walk,1)
        row_based_walk(j,i) = sum(test_digits(j,:,i));
    end
end
%% 3.b 
% Perpare datas
training_set = row_based_walk(:,1:5000);
training_labels = test_labels(1:5000,:);
testing_set = row_based_walk(:,5001:10000);
testing_labels = test_labels(5001:10000,:); 
testset = test_digits(:,:,5001:10000);
% model specification
mu = zeros(28,10);
sigma = zeros(28,10);
for i = 0:9
    mu(:,i+1)= mean(training_set(:,training_labels==i),2);
    sigma(:,i+1) = std(training_set(:,training_labels==i)')';
end
% Create model
gaussian = @(x,mu,sigma) (1/(sqrt(2*pi)*sigma))*exp(-(x-mu)^2/(2*sigma^2));
% Test the model
classification = zeros(size(testset,3),1);
max_likelihood = zeros(size(testset,3),1);
for i = 1:size(testing_set,2)
    % Calculate the testset's mu,std by row
    mu_test = (testing_set(:,i));
    likelihood = ones(1,10); % the index of maximum value is the prediction outcome
    % Put into 10HMM model
    for j = 1:10
        for k = 1:size(mu_test,1)% 28
            p_xn_mu_sigma = gaussian(mu_test(k),mu(k,j),sigma(k,j));
            if(isnan(p_xn_mu_sigma))
                p_xn_mu_sigma = 1;
            end
            likelihood(j) = likelihood(j)*p_xn_mu_sigma;
        end
    end
    % Take the maxmimum likelihood as the predicted result of the 10HMM model 
    [max_likelihood(i), index] = max(likelihood);
    classification(i) = index - 1; % Change the from index 1~10 to label 0~9
end

result = [testing_labels classification];
confusion_matrix = zeros(10,10);
for i = 1:size(result,1)
    truth = result(i,1); predicted = result(i,2);
    confusion_matrix(predicted+1,truth+1) = confusion_matrix(predicted+1,truth+1)+1;
end

figure 
plotConfMat(confusion_matrix,[0 1 2 3 4 5 6 7 8 9]);


function plotConfMat(varargin)
%PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
%   Arguments
%   - confmat:            a square confusion matrix
%   - labels (optional):  vector of class labels
switch (nargin)
    case 0
       confmat = 1;
       labels = {'1'};
    case 1
       confmat = varargin{1};
       labels = 1:size(confmat, 1);
    otherwise
       confmat = varargin{1};
       labels = varargin{2};
end
confmat(isnan(confmat))=0; % in case there are NaN elements
numlabels = size(confmat, 1); % number of labels
% calculate the percentage accuracies
confpercent = 100*confmat./repmat(sum(confmat, 1),numlabels,1);
% plotting the colors
imagesc(confpercent);
title(sprintf('Accuracy: %.2f%%', 100*trace(confmat)/sum(confmat(:))));
ylabel('Model Predicted Class'); xlabel('Groundtruth Label Class');
% set the colormap
colormap(flipud(gray));
% Create strings from the matrix values and remove spaces
textStrings = num2str([confpercent(:), confmat(:)], '%.1f%%\n%d\n');
textStrings = strtrim(cellstr(textStrings));
% Create x and y coordinates for the strings and plot them
[x,y] = meshgrid(1:numlabels);
hStrings = text(x(:),y(:),textStrings(:), ...
    'HorizontalAlignment','center');
% Get the middle value of the color range
midValue = mean(get(gca,'CLim'));
% Choose white or black for the text color of the strings so
% they can be easily seen over the background color
textColors = repmat(confpercent(:) > midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));
% Setting the axis labels
set(gca,'XTick',1:numlabels,...
    'XTickLabel',labels,...
    'YTick',1:numlabels,...
    'YTickLabel',labels,...
    'TickLength',[0 0]);
end