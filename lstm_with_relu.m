% model3_lstm_relu.m

% Define LSTM network architecture for Model 3
layers = [
    sequenceInputLayer(size(XTrain, 2))
    lstmLayer(128, 'OutputMode', 'sequence')
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

% Training options for Model 3 with batch training, mini-batches, and regularization
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

% Train Model 3
net = trainNetwork(XTrain', YTrain', layers, options);

% Save the trained Model 3
save('model3_lstm_relu.mat', 'net');
