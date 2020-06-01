function [C_new, mu, i] = k_means(X, k)
    mu = rand_init_mu(X, k);
    max_iter = 20;
    C_old  = [];
    for i = 1:max_iter
        C_new = get_C(X, mu); 
        mu    = recalculateCenters(X, C_new);

        isConverged = check_convergence(C_old, C_new);
        if isConverged == true
            return
        else
            C_old = C_new;
        end
    end
end


function bool = check_convergence(C_old, C_new)
    bool = true;
    if isempty(C_old) == true
        bool = false;
        return
    end
    [~, n] = size(C_old);
    for i = 1:n
        if C_old(i) ~= C_new(i)
            bool = false;
            break
        end
    end
end

function mu = rand_init_mu(X, k)
    [n, ~] = size(X);
    perm = randperm(n, k);
    mu = [];
    for i = 1:k
        mu = [mu; X(perm(i), :)];
    end
end

function C = get_C(X, mu)
  C = [];
    [k, ~] = size(mu);
    [n, ~] = size(X);
    for i = 1:n
        min = 999999999;
        label = 1;
        for j = 1:k
            temp = X(i, :) - mu(j, :);
            dis = sqrt(temp * temp');
            if dis < min
                min = dis;
                label = j;
            end
        end
        C = [C, label];
    end
end


function mu = recalculateCenters(X, C)
    k = length(unique(C));
    [n, d] = size(X);
    mu = [];
    for j = 1:k
        total = zeros(1, d);
        counter = 0;
        for i = 1:n
            if C(i) == j
                total = total + X(i, :);
                counter = counter + 1;
            end
        end
        mu = [mu; total/counter];
    end
end
