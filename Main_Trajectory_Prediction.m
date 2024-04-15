%% Load data from CSV file
data = readtable('Autonoumous_Car_Data.csv');
%% Calculate distance between consecutive points using Haversine formula
R = 6371; % Earth's radius in kilometers
data.Distance = NaN(height(data), 1);
for i = 1:(height(data)-1)
    lat1 = deg2rad(data.Latitude(i));
    lat2 = deg2rad(data.Latitude(i+1));
    lon1 = deg2rad(data.Longitude(i));
    lon2 = deg2rad(data.Longitude(i+1));
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    data.Distance(i) = R * c;  % Calculate distance and assign to the table
end


%% Drop the last row with NaN distance before outlier removal
data(end, :) = [];

%% Outlier removal based on IQR
Q1 = quantile(data{:, {'Latitude', 'Longitude', 'heading', 'v'}}, 0.25);
Q3 = quantile(data{:, {'Latitude', 'Longitude', 'heading', 'v'}}, 0.75);
IQR = Q3 - Q1;
lowerBound = Q1 - 1.5 * IQR;
upperBound = Q3 + 1.5 * IQR;
outliers = (data{:, {'Latitude', 'Longitude', 'heading', 'v'}} < lowerBound) | (data{:, {'Latitude', 'Longitude', 'heading', 'v'}} > upperBound);
data(outliers(:, 1) | outliers(:, 2) | outliers(:, 3) | outliers(:, 4), :) = [];


%% Normalize features and target
features = data{:, {'Latitude', 'Longitude', 'heading', 'v'}};
target = data.Distance;
[features, featureMu, featureSigma] = zscore(features);
target = (target - mean(target)) / std(target);

%% Split Data into Training, Validation, and Test Sets
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

numSamples = size(features, 1);
numTrainSamples = floor(trainRatio * numSamples);
numValSamples = floor(valRatio * numSamples);
numTestSamples = numSamples - numTrainSamples - numValSamples;

XTrain = features(1:numTrainSamples, :);
YTrain = target(1:numTrainSamples);
XVal = features(numTrainSamples+1:numTrainSamples+numValSamples, :);
YVal = target(numTrainSamples+1:numTrainSamples+numValSamples);
XTest = features(numTrainSamples+numValSamples+1:end, :);
YTest = target(numTrainSamples+numValSamples+1:end);

% Load trained models
model1 = load('model1_lstm_tanh.mat');
model2 = load('model2_lstm_relu.mat');
model3 = load('model3_lstm_relu.mat');
model4 = load('model4_gru_relu.mat');
model5 = load('model5_lstm_tanh_relu.mat')

% Evaluate each model on the test set
YPred1 = predict(model1.net, XTest');
YPred2 = predict(model2.net, XTest');
YPred3 = predict(model3.net, XTest');
YPred4 = predict(model4.net, XTest');
YPred5 = predict(model5.net, XTest');

% Calculate RMSE for each model
rmse1 = sqrt(mean((YPred1 - YTest').^2));
rmse2 = sqrt(mean((YPred2 - YTest').^2));
rmse3 = sqrt(mean((YPred3 - YTest').^2));
rmse4 = sqrt(mean((YPred4 - YTest').^2));
rmse5 = sqrt(mean((YPred5 - YTest').^2));

% Display RMSE for each model
fprintf('RMSE for Model 1 (LSTM with Tanh Activation): %.4f\n', rmse1);
fprintf('RMSE for Model 2 (LSTM with ReLU Activation): %.4f\n', rmse2);
fprintf('RMSE for Model 3 (LSTM with ReLU Activation): %.4f\n', rmse3);
fprintf('RMSE for Model 4 (GRU with ReLU Activation): %.4f\n', rmse4);
fprintf('RMSE for Model 5 (LSTM with Tanh And Relu Activation): %.4f\n', rmse5);

% Plot actual and predicted distances for comparison
figure;
plot(1:numel(YTest), YTest, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:numel(YPred1), YPred1, 'r--', 'LineWidth', 1.5);
plot(1:numel(YPred2), YPred2, 'g--', 'LineWidth', 1.5);
plot(1:numel(YPred3), YPred3, 'c--', 'LineWidth', 1.5);
plot(1:numel(YPred4), YPred4, 'm--', 'LineWidth', 1.5);
plot(1:numel(YPred5), YPred5, 'b--', 'LineWidth', 1.5);
title('Actual vs. Predicted Distances for Model Comparison');
xlabel('Sample');
ylabel('Distance (km)');
legend('Actual', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5');
grid on;



% Save training, validation, and test sets as CSV files
trainData = array2table([YTrain XTrain], 'VariableNames', {'Distance', 'Latitude', 'Longitude', 'Heading', 'Velocity'});
writetable(trainData, 'train_data.csv');

valData = array2table([YVal XVal], 'VariableNames', {'Distance', 'Latitude', 'Longitude', 'Heading', 'Velocity'});
writetable(valData, 'val_data.csv');

testData = array2table([YTest XTest], 'VariableNames', {'Distance', 'Latitude', 'Longitude', 'Heading', 'Velocity'});
writetable(testData, 'test_data.csv');

writetable(testData, 'test_data.csv');
