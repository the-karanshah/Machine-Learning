rng(0);
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;


[trIds, trLbs] = ml_load('dataset/bigbangtheory_v3/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('dataset/bigbangtheory_v3/test.mat', 'imIds');

tstLbs = [];
for i = 1:length(tstIds)
    tstLbs = [tstLbs; double(rand())];
end

bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
fprintf('Got bowCs');
trainD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
fprintf('Got trainD');
trainD = trainD';
testD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
fprintf('Got testD');
testD = testD';

% 3.4.6
fprintf('Question 3.4.6');

gamma = 1.4;
C = 20;
[trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma);
config = sprintf('-c %d -g %d -t 4', C, gamma);
model = svmtrain(trLbs, trainK, config);
[predict_label] = svmpredict(tstLbs, testK, model);
csvwrite('predTestLabels.csv', predict_label);

function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
   [n, ~] = size(trainD);
    trainK = [];
    for i = 1:n
        x = trainD(i, :);
        kernel_i = [];
        for j = 1:n
            y = trainD(j, :);
            kernel_ij = exp_kernel(x, y, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        trainK = [trainK; kernel_i];
    end
    trainK = [(1:n)', trainK];
    trainK = double(trainK);
    
    [m, ~] = size(testD);
    testK = [];
    for i = 1:m
        x = testD(i, :);
        kernel_i = [];
        for j = 1:n
            y = trainD(j, :);
            kernel_ij = exp_kernel(x, y, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        testK = [testK; kernel_i];
    end
    testK = [(1:m)', testK];
    testK = double(testK);
end

function kernel = exp_kernel(x, y, gamma)
    d = length(x);
    kernel = 0;
    for k = 1:d
        kernel = kernel + (x(k)- y(k))^2 / (x(k) + y(k) + eps('single'));
    end
    kernel = exp(kernel * (-1/gamma));
end