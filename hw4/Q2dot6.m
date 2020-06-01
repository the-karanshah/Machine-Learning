% question 2.6

% having more data points than features, rdf kernel will perform better!
% CSVs were sorted manually in Excel to map x and y properly
% with another attempt I have been trying to use python to normalise train
% data and export it here via csv which may perform good.
name = 'val_x.csv';
val_x = readmatrix(name);
val_x = val_x(2:end, 2:end).';

name = 'train_x.csv';
train_x = readmatrix(name);
train_x = train_x(2:end, 2:end).';

name = 'train_y.csv';
train_y = readmatrix(name);
train_y = train_y(1:end, 2:end);

name = 'val_y.csv';
val_y = readmatrix(name);
val_y = val_y(1:end, 2:end);

name = 'Test_Features.csv';
test_x = readmatrix(name);
test_x = test_x(1:end, 2:end).';

C = 13;
sigma = 7000;

nClass = 4;
fprintf('C=%d\n',C);

cSVMs = solveMultiSVM(train_x, train_y, C, nClass);
pred_y = predictionMulti(val_x, cSVMs, nClass);
acc = accuracy(val_y, pred_y);
fprintf("ACC %d", acc);
pred_y = predictionMulti(test_x, cSVMs, nClass);
writematrix(pred_y','submit.csv');

function cSVMs = solveMultiSVM(X, y, C, nClass)
    cSVMs = [];
    [n, ~] = size(y);
    for class = 1:nClass
        fprintf("Classifier: %d => ", class);
        binary_y = ones(n, 1);
        for i = 1:n
            if y(i) ~= class
                binary_y(i) = -1;
            end 
        end
        [SVs, w, b] = solveSVM(X, binary_y, C);
        cSVMs = [cSVMs, SVMs(class, SVs, w, b)];
    end
end

function [SVs, w, b] = solveSVM(X, y, C)
    [~, n] = size(X);
    f = -ones(n, 1);
    H = zeros(n, n);
    for i = 1:n
        for j = 1:n
            H(i, j) = get_kernel(X(:, i), X(:, j));
        end
    end
    H = H .* (y*y');
    Aeq = y';
    beq = 0;
    lb = zeros(n, 1);
    ub = C*ones(n,1);
    A = [];
    b = [];
    options = optimset('Display', 'off','LargeScale', 'off','MaxIter',350);
    alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, 0, options);
    
    [d, n] = size(X);
    w = zeros(d, 1);
    for i = 1:n
        w = w + alpha(i) * y(i) * get_k(X(:, i));
    end
    
    SVs = [];
    for i = 1:n
        if alpha(i) > eps('single')
            SVs = [SVs, SV(X(:, i), y(i), alpha(i))];
        end
    end
    
    [~,m] = size(SVs);
    fprintf('Found %d SVs\n', m);
    bs = [];
    for k = 1:m
        bs = [bs, SVs(k).y - get_k(SVs(k).x)' * w];
    end
    
    b = mean(bs);
    
end

function pred_x = predictionMulti(X, cSVMs, nClass)
    [~, n] = size(X);
    pred_x = [];
    for i = 1:n
        max_confidence = -99;
        pred_temp = 0;
        for j = 1:nClass
            class = cSVMs(j).class;
            SVs = cSVMs(j).SVs;
            w = cSVMs(j).w;
            b = cSVMs(j).b;
            confidence = pred_confidence(X(:, i), w, b);
            if confidence > max_confidence
                pred_temp = class;
                max_confidence = confidence;
            end
        end
        pred_x = [pred_x, pred_temp];
    end
end

function [confidence] = pred_confidence(X, w, b)
    
    [~, n] = size(X);
    confidence = 0;
    for i = 1:n
        confidence = confidence + w' * get_k(X(:, i)) + b;
    end
    
end

function [ku] = get_k(u)
    ku = u;
end
function [k_val] = get_kernel(u, v)
    %k_val = (k(u)' * k(v)); % linear kernel
    sigma = 5000;
    temp = get_k(u) - get_k(v);
    k_val = exp(-(temp' * temp)/sigma); %rdf kernel
end

function [acc] = accuracy(true_y, pred_y)
    [~, n] = size(pred_y);
    correct_count = 0;
    for i = 1:n
        if true_y(i) == pred_y(i)
            correct_count = correct_count + 1;
        end
    end
    acc = correct_count/n;
end
