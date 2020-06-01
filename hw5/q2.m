X = load('dataset/digit/digit.txt');
Y = load('dataset/digit/labels.txt');

% 2.5.1, 2.5.2
k = [2,4,6];
for i = 1:length(k)
    [C, mu, iter] = k_means(X, k(i));
    squareSum = total_within_group_sum_of_squares(X, C, mu);
    [p1, p2, p3] = pair_count_measure(Y, C);
    fprintf('*****\nK = %d: Iterations = %d\n*****\nsum of square = %d\np1 = %d\np2 = %d\np3 = %d\n', k(i), iter, squareSum, p1, p2, p3);
end

% 2.5.3, 2.5.4
rng(0);
repeatation = 10;
squareSum_list = [];
p1_list  = [];
p2_list  = [];
p3_list  = [];
for k = 1:10
    totalSquareSum = 0;
    p1_sum  = 0;
    p2_sum  = 0;
    p3_sum  = 0;
    for r = 1:repeatation   
        [C, mu, i] = k_means(X, k);
        squareSum = total_within_group_sum_of_squares(X, C, mu);
        [p1, p2, p3] = pair_count_measure(Y, C);
        
        totalSquareSum = totalSquareSum + squareSum;
        p1_sum = p1_sum + p1;
        p2_sum = p2_sum + p2;
        p3_sum = p3_sum + p3;
    end
    squareSum_list = [squareSum_list, totalSquareSum/repeatation];
    p1_list  = [p1_list,  p1_sum/repeatation];
    p2_list  = [p2_list,  p2_sum/repeatation];
    p3_list  = [p3_list,  p3_sum/repeatation];
end


csvwrite('p1.csv', p1_list');
csvwrite('p2.csv', p2_list');
csvwrite('p3.csv', p3_list');
csvwrite('squareSum.csv', squareSum_list');

function squareSum = total_within_group_sum_of_squares(X, C, mu)
    [k, ~] = size(mu);
    [n, ~] = size(X);
    squareSum = 0;
    for j = 1:k
        for i = 1:n
            if C(i) == j
                temp = X(i, :) - mu(j, :);
                squareSum = squareSum + temp * temp';
            end
        end
    end
end

function [p1, p2, p3] = pair_count_measure(Y, C)
    n = length(Y);
    same_class = 0;
    diff_class = 0;
    p1 = 0;
    p2 = 0;
    for i = 1:n
        for j = i+1:n
            if Y(i) == Y(j)
                same_class = same_class + 1;
                if C(i) == C(j)
                    p1 = p1 + 1;
                end
            else
                diff_class = diff_class + 1;
                if C(i) ~= C(j)
                    p2 = p2 + 1;
                end
            end
        end
    end
    p1 = p1/same_class;
    p2 = p2/diff_class;
    p3 = (p1+p2)/2;
end

