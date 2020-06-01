% question 3.4.4 - 3.4.5
run('C:\Users\Karan\Downloads\vlfeat-0.9.21\toolbox\vl_setup');
% having more features than data points, linear SVM or logistic classifier
% can perform better than rdf kernel so used linear SVM.
%3.4.4
Q3_4_4();

function Q3_4_4()
    fprintf("Question 3.4.4");
    [trD, trLb, ~, ~, ~, ~, trNegI, ~] = HW4_Utils.getPosAndRandomNeg();
    C = 2;

    [w, b, ~, alpha, obj] = solveSVM(trD, trLb, C);
    objs = [];
    aps = [];
    HW4_Utils.genRsltFile(w, b, 'val', 'rslt1');
    [ap, ~, ~] = HW4_Utils.cmpAP('rslt1', 'val');

    objs = [objs, obj];
    aps = [aps, ap];

    for iter = 1:50
        fprintf('Iteration: %d',iter);
        
        A = getA(trD, trNegI, alpha);
        B = getB(w, b);
        ifProgress = checkProgress(objs);
        [trD, trLb] = updateData(trD, trLb, A, B, ifProgress);
        
        [w, b, ~, alpha, obj] = solveSVM(trD, trLb, C);
        HW4_Utils.genRsltFile(w, b, 'val', 'valRslt');
        [ap, ~, ~] = HW4_Utils.cmpAP('valRslt', 'val');
        
        fprintf('\n ap for iteration %d is %d\n',iter,ap);
        fprintf('Obj difference %d\n',obj - objs(end));
        
        HW4_Utils.genRsltFile(w, b, 'test', sprintf('submit_%d_%d',iter, ap));
        
        objs = [objs, obj];
        aps = [aps, ap];
    end
end

function [w, b, svindex, alpha, obj] = solveSVM(X, y, C)

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
    
    [w, b, svindex] = compute_w_b(X, y, alpha);
    [obj] = get_obj(X, y, w, b, C);
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
    fprintf('\n Found %d SVs', m)
    
    bs = [];
    for i = 1:m
        k = svindex(i);
        bs = [bs, y(k) - get_k(X(:,k))' * w];
    end
    b = mean(bs); %y(i) - get_k(X(:,i))' * w;
    
end

function [obj] = get_obj(X, y, w, b, C)
    [~, n] = size(X);
    obj = 0.5 * (w') * w;
    for i = 1:n
        slack = 1 - y(i) * (w' * X(:, i) + b);
        if slack < 0
            slack = 0;
        end
        obj = obj + C * slack;
    end
end

function [ku] = get_k(u)
    ku = u;
end

function [k_val] = get_kernel(u, v)
    k_val = (get_k(u)' * get_k(v));
end

function [A] = getA(trD, trNegI, alpha)
   A = [];
    [~, n] = size(trD);
    for i = 1:(trNegI-1)
        A = [A, 0];
    end
    for i = trNegI:n
        if alpha(i) < eps('single')
            A = [A, 1];
        else
            A = [A, 0];
        end
    end
end

function [hnrects] = getB(w, b)
    B = []; 

    imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW4_Utils.dataDir, 'train'), 'jpg');
    nIm = length(imFiles);
    rects = cell(1, nIm);
    startT = tic;
    for i=1:nIm
        ml_progressBar(i, nIm, 'Ub detection', startT);
        im = imread(imFiles{i});
        rects{i} = HW4_Utils.detect(im, w, b);
    end

    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, 'train'), 'ubAnno');    
    nIm = length(ubAnno);
    [detScores, isTruePos] = deal(cell(1, nIm));

    for i=1:nIm
        rects_i = rects{i};
        detScores{i} = rects_i(5,:);
        ubs_i = ubAnno{i};
        isTruePos_i = -ones(1, size(rects_i, 2));
        for j=1:size(ubs_i,2)
            ub = ubs_i(:,j);
            overlap = HW4_Utils.rectOverlap(rects_i, ub);
            isTruePos_i(overlap >= 0.5) = 1;
        end
        isTruePos{i} = isTruePos_i;
    end
    
    hnrects = [];
    for i = 1:nIm
        rects_i = rects{i};
        isTruePos_i = isTruePos{i};
        [~, m] = size(rects_i);
        for j = 1:m
            if isTruePos_i(j) == -1
                slack = 1 + rects_i(5, j);
                if slack > 1
                    hn = [rects_i(1:4, j); i; slack];
                    hnrects = [hnrects, hn];
                end
            end
        end
    end
end

function ifProgress = checkProgress(objs)
    [~, n] = size(objs);
    ifProgress = 1;
    if n < 2
        ifProgress = 1;
    else
        if objs(n) > objs(n-1)
            ifProgress = 1;
        end
    end
end

function [trD, trLb] = updateData(trD, trLb, A, B, ifProgress)
    if ifProgress == 1
        trD = trD(:, ~A);
        trLb = trLb(~A, :);
    end

    im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', 1));
    [imH, imW, ~] = size(im);
    outOfBound = [];
    [~, n] = size(B);
    for i=1:n
        if B(3, i) > imW || B(4, i) > imH
            outOfBound = [outOfBound, 1];
        else
            outOfBound = [outOfBound, 0];
        end
    end
    B = B(:, ~outOfBound);
    
    m = 548;
    [~, n] = size(trD);
    diff = m - n;
    [~, n] = size(B);
    slacks = B(6, :);
    threshhold = max(slacks);
    if n > m
        slacks = sort(slacks, 'descend');
        threshhold = slacks(diff);
    end
    
    hardNegatives = [];
    for i=1:n
        im_i = B(5, i);
        im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', im_i));
        %[imH, imW, ~] = size(im);
        if B(6, i) < threshhold
            continue
        end
        imReg = im(round(B(2,i)):round(B(4,i)), round(B(1,i)):round(B(3,i)),:);
        imReg = imresize(imReg, HW4_Utils.normImSz);
        D_i = HW4_Utils.cmpFeat(rgb2gray(imReg));
        hardNegatives = [hardNegatives, D_i];
        trLb = [trLb; -1];
    end
    hardNegatives = HW4_Utils.l2Norm(double(hardNegatives));
    trD = [trD, hardNegatives];
end



