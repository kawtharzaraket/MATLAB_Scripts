% Load pre-trained ResNet-50 model
dataDir = 'trafficnet_dataset/train';  % Specify the path to your image data
imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
dataDir2 = 'trafficnet_dataset/test';  % Specify the path to your validation data
imdsValidation = imageDatastore(dataDir2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Data Augmentation (optional)
% augmenter = imageDataAugmenter(...);
% augmentedImds = augmentedImageDatastore(...);
net = inceptionresnetv2;

% Specify the layers to keep (up to 'res4a_relu')
layersToKeep = {'conv1_relu', 'pool1', 'res2a_relu', 'res2b_relu', 'res2c_relu', 'res3a_relu', 'res3b_relu', 'res3c_relu', 'res3d_relu', 'res4a_relu'};

% Find the indices of layers to remove
layersToRemoveIdx = find(~ismember({net.Layers.Name}, layersToKeep));

% Create a new network with the desired layers
newLayers = net.Layers;
newLayers(layersToRemoveIdx) = [];

% Define the number of classes (categories) in your dataset
numClasses = numel(categories(imds.Labels));

% Resize the input images to match the expected size (224x224x3)
imds = augmentedImageDatastore([224 224], imds);

% Add custom layers for classification
layers = [
    imageInputLayer([224 224 3], 'Name', 'input')
    newLayers
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];
imdsValidationResized = augmentedImageDatastore([224 224], imdsValidation);
% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', imdsValidationResized, ...
    'ValidationFrequency', 5, ...
    'Verbose', true);

[inceptionFineTuned, inceptiontrainingInfo] = trainNetwork(imds, layers, options);


YPred = classify(inceptionFineTuned, imdsValidationResized);

% Calculate accuracy
accuracy = mean(YPred == imdsValidation.Labels);
disp(['Accuracy: ' num2str(accuracy)]);

% Display the confusion matrix
figure;
plotconfusion(imdsValidation.Labels, YPred);
saveas(gcf, 'confusion_inception_updated_matrix2.png');

figure;
plot(inceptiontrainingInfo.TrainingAccuracy);
xlabel('Epoch');
ylabel('Training Accuracy');
title('Training Accuracy');
saveas(gcf, 'inception_training_accuracy2.png');
% Display the first few images along with their predicted labels
figure;
numImages = 5; % Number of images to display
for i = 1:numImages
    subplot(1, numImages, i);
    imshow(imdsValidation.Files{i});
    title(['Predicted: ' char(YPred(i))]); % Assuming YPred contains class names
end
saveas(gcf, 'updated_inception_sample_images_with_labels2.png');
