% Step 1: Data Preprocessing
% Preprocess your dataset as needed

% Step 2: Load the Dataset
%load('yourMatrix.mat'); % Load your preprocessed outlier dataset

% Step 3: Prepare Data for Training
% Split dataset into training and validation sets (e.g., 80% training, 20% validation)
% Ensure your dataset is in the appropriate format (matrix or table)
% Replace splitYourDataFunction with your actual splitting logic
trainingSize = 0.8 * size(newMatrix, 1);
idx = datasample(1:size(newMatrix, 1), trainingSize, 'Replace', false);
trainData = newMatrix(idx, :);
validationData = newMatrix(setdiff(1:size(newMatrix, 1), idx), :);

%[trainData, validationData] = splitYourDataFunction(yourMatrix);

% Step 4: Create the Neural Network Model
% Assuming 'IsOutlier' is your target variable (column vector)
%outlierLabels = ones(4800,1);
% Assuming 'outlierLabels' is your target variable (column vector)
%numSamples = size(outlierLabels, 1);
%binaryLabels = [outlierLabels, ~outlierLabels]; % Convert to binary labels (1 for outlier, 0 for non-outlier)
%targetMatrix = full(ind2vec(binaryLabels', 2)); % Convert to one-hot encoded matrix

%net = configure(net, trainData', targetMatrix); % Configure the network with input and output sizes

%numClasses = 2; % Number of classes (outlier and non-outlier)
%binaryLabels = [IsOutlier, 0]; % Convert to binary labels (1 for outlier, 0 for non-outlier)
%targetMatrix = full(ind2vec(binaryLabels', numClasses)); % Convert to one-hot encoded matrix

%numSamples = size(binaryLabels, 1); % Number of samples
%numClasses = 2; % Number of classes (outlier and non-outlier)

% Initialize target matrix
%targetMatrix = zeros(numSamples, numClasses);

% Set appropriate columns to 1 based on binary labels
%targetMatrix(binaryLabels == 1, 1) = 1; % Set 1st column to 1 for outlier
%targetMatrix(binaryLabels == 0, 2) = 1; % Set 2nd column to 1 for non-outlier
% Assuming 'IsOutlier' is your target variable (column vector)
outlierLabels = ones(4800, 1); % Example of outlier labels
nonOutlierLabels = zeros(4800, 1); % Example of non-outlier labels

binaryLabels = [outlierLabels, nonOutlierLabels]; % Convert to binary labels (1 for outlier, 0 for non-outlier)

numSamples = size(binaryLabels, 1);
numClasses = size(binaryLabels, 2);
targetMatrix = zeros(numSamples, numClasses);
for i = 1:numSamples
    if binaryLabels(i, 1) == 1
        targetMatrix(i, 1) = 1; % Set 1st column to 1 for outlier
    else
        targetMatrix(i, 2) = 1; % Set 2nd column to 1 for non-outlier
    end
end

hiddenLayerSize = 50; % Number of neurons in the hidden layer
net = patternnet(hiddenLayerSize); % Create a pattern recognition neural network
net = configure(net, trainData', targetMatrix'); % Configure the network with input and output sizes

[trainDataNormalized, ~] = mapminmax(trainData'); % Normalize training data
[net, ~] = train(net, trainDataNormalized, targetMatrix'); % Train the neural network

validationDataNormalized = mapminmax(validationData'); % Normalize validation data
outputValidation = net(validationDataNormalized); % Perform validation using trained network

save('trained_network.mat', 'net');

