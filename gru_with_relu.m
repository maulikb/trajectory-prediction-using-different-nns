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
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XVal', YVal'}, ...
    'ValidationFrequency', 2, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.9, ...
    'L2Regularization', 0.01, ...
    'OutputFcn', @stopIfValidationRMSEIncreases4); % Use the modified output function

% Train Model 4
net = trainNetwork(XTrain', YTrain', layers, options);

% Save the trained Model 4
save('model4_gru_relu.mat', 'net');


% Save the trained Model 4
save('model4_gru_relu.mat', 'net');




% Output function for Model 4
function stop = stopIfValidationRMSEIncreases4(info)
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
