% model1_lstm_tanh.m

% Define LSTM network architecture for Model 1
layers = [
    sequenceInputLayer(size(XTrain, 2))
    lstmLayer(128, 'OutputMode', 'sequence')
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(256)
    tanhLayer
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XVal', YVal'}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.9, ...
    'L2Regularization', 0.01, ...
    'OutputFcn', @stopIfValidationRMSEIncreases); % Use the modified output function




% Train Model 1
net = trainNetwork(XTrain', YTrain', layers, options);

% Save the trained Model 1
save('model1_lstm_tanh.mat', 'net');


function stop = stopIfValidationRMSEIncreases(info)
    persistent bestValRMSE numIncreases
    stop = false; % This function should not issue a stop command

    % Initialize the bestValRMSE and numIncreases
    if info.State == "start"
        bestValRMSE = inf;
        numIncreases = 0;
    end

    % Update the bestValRMSE and numIncreases each iteration
    if info.State == "iteration" && ~isempty(info.ValidationRMSE)
        % If the RMSE has increased by a small threshold, count it as an increase
        if info.ValidationRMSE > bestValRMSE + 0.01  % smaller threshold than before
            numIncreases = numIncreases + 1;
        else
            bestValRMSE = info.ValidationRMSE;
            numIncreases = 0; % reset counter
        end
        % Stop if there have been 3 consecutive increases
        if numIncreases >= 3 % more sensitive to increases
            stop = true;
        end
    end
end

