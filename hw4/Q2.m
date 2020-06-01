% question 2.2 to 2.5

q2_1_data = load('data\q2_1_data.mat');
trD  = q2_1_data.trD;
trLb  = q2_1_data.trLb;
valD  = q2_1_data.valD;
valLb  = q2_1_data.valLb;

C1 = 0.1;
C2 = 10;

Q2_4(trD, trLb, valD, valLb, C1);
Q2_4(trD, trLb, valD, valLb, C2);

function Q2_4(trD, trLb, valD, valLb, C)
    [w, b, svindex, obj] = solveSVM(trD, trLb, C);
    
    predLb = prediction(valD, w, b);
    acc = accuracy(valLb, predLb);
    fprintf('\n C = %d => Accuracy %d', C, acc)
        
    fprintf('\n Objective Value of SVM= %d', obj);
    
    [~,m] = size(svindex);
    fprintf('\n Number of Support Vectors %d', m);
    
    trueLb = get_tranLabel(valLb');
    predLb = get_tranLabel(predLb);
    plotconfusion(trueLb, predLb);
end

function [w, b, svindex, obj] = solveSVM(X, y, C)
    [~, n] = size(X);    
    f = -ones(n, 1);
    H = get_kernel(X, X) .* (y*y');
    Aeq = y';            
    beq = 0;
    lb = zeros(n, 1);  
    ub(1:n, 1:1) = C;
    A = [];
    b = [];
    options = optimset('Display', 'off','LargeScale', 'off','MaxIter',350);
    alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, 0, options);
    svindex = [];
    for i = 1:n
        if alpha(i) > 0
            svindex = [svindex, i];
        end
    end
    [w, b, svindex] = compute_w_b(X, y, alpha);
    obj = get_objective(X, y, alpha);
end

function [w, b, svindex] = compute_w_b(X, y, alpha)
    [d, n] = size(X);
    w = zeros(d, 1);
    for i = 1:n
        w = w + alpha(i) * y(i) * get_k(X(:,i));
    end

    svindex = [];
    for i = 1:n
        if alpha(i) > eps('single')
            svindex = [svindex, i];
        end
    end
    
    [~,m] = size(svindex);
    %fprintf('\n got %d SVs', m)
    bs = [];
    for i = 1:m
        k = svindex(i);
        bs = [bs, y(k) - get_k(X(:,k))' * w];
    end
    
    b = mean(bs);
end

function pred = prediction(X, w, b)
    [~, n] = size(X);
    pred = [];
    for i = 1:n
        y = w' * get_k(X(:, i)) + b;
        if y > 0
            pred = [pred, 1];
        else
            pred = [pred, -1];
        end
    end
end

function [ku] = get_k(u)
    ku = u;
end

function [k_val] = get_kernel(u, v)
    k_val = (get_k(u)' * get_k(v)); %linear kernel
end

function acc = accuracy(y, y_pred)
    [~, n] = size(y_pred);
    counter = 0;
    for i = 1:n
        if y(i) == y_pred(i)
            counter = counter + 1;
        end
    end
    acc = counter/n;
end

function obj = get_objective(X, y, alpha)
    [~, n] = size(X);
    f = ones(n, 1);
    H = get_kernel(X, X) .* (y*y');
    obj = f'*alpha - 0.5 * alpha' * H * alpha;
end

function transLb = get_tranLabel(y)
    transLb = [];
    [~, n] = size(y);
    for i = 1:n
        if y(i) == -1
            transLb = [transLb, 0];
        else
            transLb = [transLb, 1];
        end
    end
end