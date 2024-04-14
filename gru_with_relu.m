% model4_gru_relu.m

% Define GRU network architecture for Model 4
layers = [
    sequenceInputLayer(size(XTrain, 2))
    gruLayer(128)
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

% Training options for Model 4 with batch training, mini-batches, and regularization
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XVal', YVal'}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.9, ...
    'L2Regularization', 0.01);

% Train Model 4
net = trainNetwork(XTrain', YTrain', layers, options);

% Save the trained Model 4
save('model4_gru_relu.mat', 'net');
