clc;clear;close all;
% Get YOLO detector
set(0,'defaultfigurecolor','w') 
preTrainedDetector = yolov3ObjectDetector('tiny-yolov3-coco');

% Define the groundtruth boxes for image coco2
coco2GTboxesXYXY = [103 238 228 317;
                    283 219 373 294;
                    352 228 439 294;
                    447 227 520 285;
                    469 224 553 283;
                    575 218 632 262];
coco2GTboxesXYWH = coco2GTboxesXYXY;
coco2GTboxesXYWH(:, 3) = coco2GTboxesXYXY(:, 3) - coco2GTboxesXYXY(:, 1);
coco2GTboxesXYWH(:, 4) = coco2GTboxesXYXY(:, 4) - coco2GTboxesXYXY(:, 2);
%%
% Read image and detect objects
testImage = imread(".\coco2.jpg");
[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage, SelectStrongest=false, Threshold=0.2); % Enabling SelectStrongest applies non-maximum suppression.
% Show the original prediction without the IOU threhold
predictedScoreVisualization_0 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
predictedLabelVisualization_0 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);

%
[predictedBboxes,predictedScores,predictedLabels] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,coco2GTboxesXYWH,0.1);
groundTruthVisualization =    insertObjectAnnotation(testImage,'rectangle', coco2GTboxesXYWH, "zebra");
predictedScoreVisualization_1 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
predictedLabelVisualization = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);


[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage, SelectStrongest=false, Threshold=0.5); % Enabling SelectStrongest applies non-maximum suppression.
[predictedBboxes,predictedScores,predictedLabels] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,coco2GTboxesXYWH,0.3);
groundTruthVisualization =    insertObjectAnnotation(testImage,'rectangle', coco2GTboxesXYWH, "zebra");
predictedScoreVisualization_2 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
predictedLabelVisualization = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);

[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage, SelectStrongest=false, Threshold=0.5); % Enabling SelectStrongest applies non-maximum suppression.
[predictedBboxes,predictedScores,predictedLabels] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,coco2GTboxesXYWH,0.5);
groundTruthVisualization =    insertObjectAnnotation(testImage,'rectangle', coco2GTboxesXYWH, "zebra");
predictedScoreVisualization_3 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
predictedLabelVisualization = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);

[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage, SelectStrongest=false, Threshold=0.5); % Enabling SelectStrongest applies non-maximum suppression.
[predictedBboxes,predictedScores,predictedLabels] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,coco2GTboxesXYWH,0.7);
predictedScoreVisualization =    insertObjectAnnotation(testImage,'rectangle', coco2GTboxesXYWH, "zebra");
predictedScoreVisualization_4 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
predictedLabelVisualization = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);

[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage, SelectStrongest=false, Threshold=0.5); % Enabling SelectStrongest applies non-maximum suppression.
[predictedBboxes,predictedScores,predictedLabels] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,coco2GTboxesXYWH,0.9);

if(isempty(predictedLabels)==0)
    groundTruthVisualization =    insertObjectAnnotation(testImage,'rectangle', coco2GTboxesXYWH, "zebra");
    predictedScoreVisualization_5 = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedScores);
    predictedLabelVisualization = insertObjectAnnotation(testImage,'rectangle',predictedBboxes,predictedLabels);
else
    % Just show original image
    predictedScoreVisualization_5 = testImage;
end

%Visualize
f = figure(1);
f.Position = [0,0, 1920, 1080];
set(gca, 'LooseInset', get(gca,'TightInset'))
axis tight
subplot(2,3,1)
imshow(predictedScoreVisualization_0)
title('Prediction without IOU threshold')
subplot(2,3,2)
imshow(predictedScoreVisualization_1) % Feel free to replace with predictedLabelVisualization
title('YOLO predicted boxes with scores (IOU = 0.1)')
subplot(2,3,3)
imshow(predictedScoreVisualization_2) % Feel free to replace with predictedLabelVisualization
title('IOU = 0.3')
subplot(2,3,4)
imshow(predictedScoreVisualization_3) % Feel free to replace with predictedLabelVisualization
title('IOU = 0.5')
subplot(2,3,5)
imshow(predictedScoreVisualization_4) % Feel free to replace with predictedLabelVisualization
title('IOU = 0.7')
subplot(2,3,6)
imshow(predictedScoreVisualization_5) % Feel free to replace with predictedLabelVisualization
title('IOU = 0.9')

figure
imshow(predictedScoreVisualization)
f = figure(1);
f.Position = [0,0, 1920, 1080];
set(gca, 'LooseInset', get(gca,'TightInset'))
axis tight
% imshow(imread('coco2.jpg'))
% for i = 1:size(predictedBboxes)
%     h = drawrectangle('Position',predictedBboxes(i,:),'StripeColor','r');
% end
% for i = 1:size(coco2GTboxesXYWH)
%     h = drawrectangle('Position',coco2GTboxesXYWH(i,:),'StripeColor','y');
% end
%% 4.b
set(0,'defaultfigurecolor','w') 
preTrainedDetector = yolov3ObjectDetector('tiny-yolov3-coco');
testImage1 = imread(".\coco1.jpg");
testImage2 = imread(".\coco2.jpg");
testImage3 = imread(".\coco3.jpg");
% Original box
[predictedBboxes1,predictedScores1,predictedLabels1] = detect(preTrainedDetector, testImage1, SelectStrongest=false, Threshold=0.5);
wider_boxes1 = Animal_is_important(predictedBboxes1,predictedLabels1); % return the wider boxes
predictedScoreVisualization_1 = insertObjectAnnotation(testImage1,'rectangle',predictedBboxes1,predictedLabels1,'Color',{'yellow'});
%
[predictedBboxes2,predictedScores2,predictedLabels2] = detect(preTrainedDetector, testImage2, SelectStrongest=false, Threshold=0.5);
wider_boxes2 = Animal_is_important(predictedBboxes2,predictedLabels2);
predictedScoreVisualization_2 = insertObjectAnnotation(testImage2,'rectangle',predictedBboxes2,predictedLabels2,'Color',{'yellow'});
%
[predictedBboxes3,predictedScores3,predictedLabels3] = detect(preTrainedDetector, testImage3, SelectStrongest=false, Threshold=0.5);
wider_boxes3 = Animal_is_important(predictedBboxes3,predictedLabels3);
predictedScoreVisualization_3 = insertObjectAnnotation(testImage3,'rectangle',predictedBboxes3,predictedLabels3,'Color',{'yellow'});

% Plot, original bbox and new bbox
figure
subplot(1,3,1)
imshow(predictedScoreVisualization_1)
title('coco1.jpg, red: wider boxes, yellow: original boxes')
for i = 1:size(wider_boxes1)
    h = rectangle('Position',wider_boxes1(i,:),'EdgeColor','r','LineWidth',1);
end
subplot(1,3,2)
imshow(predictedScoreVisualization_2)
title('coco2.jpg, red: wider boxes, yellow: original boxes')
for i = 1:size(wider_boxes2)
    h = rectangle('Position',wider_boxes2(i,:),'EdgeColor','r','LineWidth',1);
end
subplot(1,3,3)
imshow(predictedScoreVisualization_3)
title('coco3.jpg, red: wider boxes, yellow: original boxes')
for i = 1:size(wider_boxes3)
    h = rectangle('Position',wider_boxes3(i,:),'EdgeColor','r','LineWidth',1);
end
%% 4.c 
testImage1 = imread(".\coco1.jpg");
testImage2 = imread(".\coco2.jpg");
testImage3 = imread(".\coco3.jpg");
% image 1
[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage1, SelectStrongest=false, Threshold=0.2);% This is T thershold here
predictedLabelsVisualization_original1 = insertObjectAnnotation(testImage1,'rectangle',predictedBboxes,predictedLabels);
[predictedBboxes_suppresion, predictedBboxes_labels_nms] = non_max_suppression(predictedBboxes,predictedLabels,0.2);
predictedLabelsVisualization_nms1 = insertObjectAnnotation(testImage1,'rectangle',predictedBboxes_suppresion,predictedBboxes_labels_nms);
% image 2
[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage2, SelectStrongest=false, Threshold=0.2);% This is T thershold here
predictedLabelsVisualization_original2 = insertObjectAnnotation(testImage2,'rectangle',predictedBboxes,predictedLabels);
[predictedBboxes_suppresion, predictedBboxes_labels_nms] = non_max_suppression(predictedBboxes,predictedLabels,0.2);
predictedLabelsVisualization_nms2 = insertObjectAnnotation(testImage2,'rectangle',predictedBboxes_suppresion,predictedBboxes_labels_nms);
% image 3
[predictedBboxes,predictedScores,predictedLabels] = detect(preTrainedDetector, testImage3, SelectStrongest=false, Threshold=0.2);% This is T thershold here
predictedLabelsVisualization_original3 = insertObjectAnnotation(testImage3,'rectangle',predictedBboxes,predictedLabels);
[predictedBboxes_suppresion, predictedBboxes_labels_nms] = non_max_suppression(predictedBboxes,predictedLabels,0.2);
predictedLabelsVisualization_nms3 = insertObjectAnnotation(testImage3,'rectangle',predictedBboxes_suppresion,predictedBboxes_labels_nms);

figure
subplot(1,2,1)
imshow(predictedLabelsVisualization_original1)
title('Before non-maximum suppression')
subplot(1,2,2)
imshow(predictedLabelsVisualization_nms1)
title('After non-maximum suppression')

figure
subplot(1,2,1)
imshow(predictedLabelsVisualization_original2)
title('Before non-maximum suppression')
subplot(1,2,2)
imshow(predictedLabelsVisualization_nms2)
title('After non-maximum suppression')

figure
subplot(1,2,1)
imshow(predictedLabelsVisualization_original3)
title('Before non-maximum suppression')
subplot(1,2,2)
imshow(predictedLabelsVisualization_nms3)
title('After non-maximum suppression')
%% Thresholding the IOU
function [return_boxes,new_score,new_label] = threshold_IOU(predictedBboxes,predictedScores,predictedLabels,ground_true,thres)
    return_boxes = zeros(size(predictedBboxes));
    new_label = strings(size(predictedLabels));
    new_score = zeros(size(predictedScores));
    for i = 1:size(predictedBboxes,1)
         p_x = predictedBboxes(i,1);
         p_y = predictedBboxes(i,2);
         p_w = predictedBboxes(i,3);
         p_h = predictedBboxes(i,4);
         p_x2 = p_x+p_w;
         p_y2 = p_y+p_h;
        for j = 1:size(ground_true,1)
             t_x = ground_true(j,1);
             t_y = ground_true(j,2);
             t_w = ground_true(j,3);
             t_h = ground_true(j,4);
             t_x2 = t_x+t_w;
             t_y2 = t_y+t_h;
             ixmin = max(t_x, p_x);
             ixmax = min(t_x2, p_x2);
             iymin = max(t_y, p_y);
             iymax = min(t_y2, p_y2);
             iw = ixmax-ixmin;
             ih = iymax-iymin;
             %
             flag1 = t_x - p_x; 
             flag2 = t_x - p_x2;
             flag3 = t_x2 - p_x; 
             flag4 = t_x2 - p_x;
             if((flag1>=0 && flag2>=0 && flag3>=0 && flag4>=0) || (flag1<=0 && flag2<=0 && flag3<=0 && flag4<=0))
                 X_intersect_flag = 0;
             else
                 X_intersect_flag = 1;
             end
             %
             flag1 = t_y - p_y; 
             flag2 = t_y - p_y2;
             flag3 = t_y2 - p_y; 
             flag4 = t_y2 - p_y;
             if((flag1>=0 && flag2>=0 && flag3>=0 && flag4>=0) || (flag1<=0 && flag2<=0 && flag3<=0 && flag4<=0))
                 Y_intersect_flag = 0;
             else
                 Y_intersect_flag = 1;
             end
             %
             if (X_intersect_flag+Y_intersect_flag~=0)
                inters = iw*ih;
             else
                inters = 0;
             end
             inters = rectint([p_x p_y p_w p_h],[t_x t_y t_w t_h]);
             uion = t_w*t_h+p_h*p_w-inters;
             iou = inters/uion;
             if (iou >= thres)
                return_boxes(i,1) = p_x;
                return_boxes(i,2) = p_y;
                return_boxes(i,3) = p_w;
                return_boxes(i,4) = p_h;
                new_label(i) = predictedLabels(i);
                new_score(i) = predictedScores(i);
             end
        end
    end
    new_label = new_label(find(~cellfun(@isempty,new_label)))';
    return_boxes = nonzeros(return_boxes);
    new_score = nonzeros(new_score);
    return_boxes = reshape(return_boxes,[],4);
end
%% Function of 4.b
function [wider_boxes] = Animal_is_important(predictedBboxes,predictedLabels)
%automatically changes the output of the network 
% so that all bounding boxes of animals are 20% wider,
%but keep the box size of other classes such as cars the same. 
% Do not change the box height.
% for these three image: they have: zebra, elephant, person (is also animal??)
    wider_boxes = predictedBboxes;
    for i = 1:1:size(predictedLabels)
        if((string(predictedLabels(i)) == "zebra") || (string(predictedLabels(i)) == "elephant") || (string(predictedLabels(i)) == "person"))
            wider_boxes(i,3) = wider_boxes(i,3)*1.2; % wide of image 
            wider_boxes(i,1) = wider_boxes(i,1) - 0.1*wider_boxes(i,3); % wider the box from the center, instead from the up-left corner
        end
    end
end
%
function [predictedBboxes,labels]= non_max_suppression(predictedBboxes,labels,T)
    for i = 1:size(labels)
        [predictedBboxes,labels]= non_max_suppression_loop(predictedBboxes,labels,T);
    end
end
function [nms_bbox,nms_label]= non_max_suppression_loop(predictedBboxes,labels,T)
    % First find the all correspoding bbox for the each ground truth
    % Keep the fist small bbox and the higest score bbox
    % Second, discard the all the bbox except the higest score 
    nms_bbox = [];
    nms_label = [];
    nms_score = [];
    num_of_tpye = unique(labels);
    % Do NMS By label
    for i = 1:size(num_of_tpye,1)
        % Index list
        same_type_object = predictedBboxes(find(labels==num_of_tpye(i)),:);
        selected_list = zeros(size(same_type_object,1),1);
        % Find the iou between each other, select iou > 0.5 as them
        % Boxed that selected same object
        for j = 1:size(same_type_object,1)
            if(selected_list(j) == 0) % If no been selected
                selected_list(j) = 1;
            else 
                continue;
            end
            for k = j+1:size(same_type_object,1)
                iou = cal_IOU(same_type_object(j,:),same_type_object(k,:));
                if((iou > 0) && (iou~=1) && selected_list(k) ~= -1) % with overlap
                    selected_list(k)=1;
                end
            end
            % On the list, select the same object, and find the higest
            % score
            overlap_obj = same_type_object(find(selected_list~=0 & selected_list~=-1),:);
            % Find the max box the cover them, aka, find the boundary of
            % all prediction
            x_min = inf; x_max = -inf; y_min = inf; y_max =-inf;
            for h = 1:size(overlap_obj,1)
                if(overlap_obj(h,1)<x_min)
                    x_min = overlap_obj(h,1);
                end
                if(overlap_obj(h,2)<y_min)
                    y_min = overlap_obj(h,2);
                end
                if(overlap_obj(h,1)+overlap_obj(h,3)>x_max)
                    x_max = overlap_obj(h,1) + overlap_obj(h,3);
                end
                if(overlap_obj(h,2)+overlap_obj(h,4)>y_max)
                    y_max = overlap_obj(h,2)+overlap_obj(h,4);
                end
            end
            nms_bbox = [nms_bbox; [x_min y_min x_max-x_min y_max-y_min]];
            nms_label = [nms_label; num_of_tpye(i)];
            selected_list(find(selected_list==1)) = -1;
        end
    end
end
%% calculate IOU every callback
function [iou ]= cal_IOU(predictedBboxes,ground_true)
         p_x = predictedBboxes(1);
         p_y = predictedBboxes(2);
         p_w = predictedBboxes(3);
         p_h = predictedBboxes(4);
         p_x2 = p_x+p_w;
         p_y2 = p_y+p_h;
         t_x = ground_true(1);
         t_y = ground_true(2);
         t_w = ground_true(3);
         t_h = ground_true(4);
         t_x2 = t_x+t_w;
         t_y2 = t_y+t_h;
         inters = rectint([p_x p_y p_w p_h],[t_x t_y t_w t_h]);
         uion = t_w*t_h+p_h*p_w-inters;
         iou = inters/uion;
end