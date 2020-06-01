rng(0);
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;


[trIds, trLbs] = ml_load('dataset/bigbangtheory_v3/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('dataset/bigbangtheory_v3/test.mat', 'imIds'); 
fprintf('Dataset loaded\n');

bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
fprintf('Got bowCs\n');
%save('bowCs.mat', 'bowCs');

trainD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
fprintf('Got trainD\n');
trainD = trainD';
%save('trainD.mat', 'trainD');


% 3.4.2
fprintf('Question 3.4.2\n');
acc = HW5_BoW.main();

% 3.4.5

C = [30, 50, 100, 200];
gamma = [1.1, 1.4, 2.0];
folds = 5;
fprintf('Question 3.4.5\n');
for j = 1:length(gamma)
    [trainK] = cmpExpX2Kernel(trainD, gamma(j));
    for i = 1:length(C)
        config = sprintf('-c %d -g %d -t 4 -v %d -q', C(i), gamma(j), folds);
        acc = svmtrain(trLbs, trainK, config);
        fprintf('C = %d, gamma = %d\n', C(i), gamma(j));
        fprintf('****\n');
    end
    fprintf('**********\n');
end

function [trainK] = cmpExpX2Kernel(trainD, gamma)
    [n, ~] = size(trainD);
    trainK = [];
    for i = 1:n
        kernel_i = [];
        for j = 1:n
            kernel_ij = exp_kernel(trainD, i, j, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        trainK = [trainK; kernel_i];
    end
    trainK = [(1:n)', trainK];
    trainK = double(trainK);
end


function kernel = exp_kernel(X, i, j, gamma)
    x = X(i, :);
    y = X(j, :);
    d = length(x);
    kernel = 0;
    for k = 1:d
        kernel = kernel + (x(k)- y(k))^2 / (x(k) + y(k) + eps('single'));
    end
    kernel = exp(kernel * (-1/gamma));
end

