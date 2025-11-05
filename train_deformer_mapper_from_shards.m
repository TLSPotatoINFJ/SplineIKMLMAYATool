% train_deformer_mapper_from_npz.m
% 读取一个目录下的一堆单样本 .npz（与 Maya 工具导出的结构一致），训练一个 MLP，
% 并导出 onnx + prep.json + selfcheck_bundle.npz
%
% 用法：
%   train_deformer_mapper_from_npz("D:\WKS\SamplesSingle", "D:\WKS\Models\deformer_mapper");

function train_deformer_mapper_from_npz(sampleDir, outDir)

    arguments
        sampleDir (1,:) char
        outDir    (1,:) char
    end

    if ~isfolder(sampleDir)
        error("样本目录不存在: %s", sampleDir);
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    % ------------------------------------------------------------
    % 1. 扫目录，把所有 .npz 读出来
    % ------------------------------------------------------------
    files = dir(fullfile(sampleDir, '*.npz'));
    if isempty(files)
        error("目录 %s 下没有 .npz 样本。", sampleDir);
    end

    fprintf("Found %d npz samples.\n", numel(files));

    Xcells = {};
    Ycells = {};
    flatOrder = "row";  % 默认

    for i = 1:numel(files)
        fpath = fullfile(files(i).folder, files(i).name);
        [x60, y36, fo] = read_single_npz(fpath);
        Xcells{end+1} = x60(:); %#ok<AGROW>
        Ycells{end+1} = y36(:); %#ok<AGROW>
        if i == 1
            flatOrder = fo;
        end
    end

    % 拼成矩阵：X:(60,N)  Y:(36,N)
    X = cat(2, Xcells{:});
    Y = cat(2, Ycells{:});
    N = size(X,2);

    fprintf("Loaded %d samples. X: %dx%d, Y: %dx%d\n", N, size(X,1), size(X,2), size(Y,1), size(Y,2));

    % ------------------------------------------------------------
    % 2. 划分训练/验证
    % ------------------------------------------------------------
    idx = randperm(N);
    numVal = max(1, floor(0.1 * N));
    valIdx = idx(1:numVal);
    trnIdx = idx(numVal+1:end);

    Xtr = X(:, trnIdx);
    Ytr = Y(:, trnIdx);
    Xva = X(:, valIdx);
    Yva = Y(:, valIdx);

    % dlarray 形式
    XtrDL = dlarray(single(Xtr), 'CB');  % (C,B)
    YtrDL = dlarray(single(Ytr), 'CB');

    % ------------------------------------------------------------
    % 3. 建网络（普通 MLP）
    % ------------------------------------------------------------
    net = build_simple_mlp();

    % 把 dlnetwork 包起来
    dlnet = dlnetwork(net);

    % ------------------------------------------------------------
    % 4. 训练
    % ------------------------------------------------------------
    numEpochs = 200;
    batchSize = 256;
    learnRate = 1e-3;

    trailingAvg = [];
    trailingAvgSq = [];

    iteration = 0;
    numBatches = ceil(size(Xtr,2) / batchSize);

    for epoch = 1:numEpochs
        % 打乱
        order = randperm(size(Xtr,2));
        Xtr = Xtr(:, order);
        Ytr = Ytr(:, order);

        for b = 1:numBatches
            iteration = iteration + 1;

            idxStart = (b-1)*batchSize + 1;
            idxEnd   = min(b*batchSize, size(Xtr,2));
            xb = dlarray(single(Xtr(:, idxStart:idxEnd)), 'CB');
            yb = dlarray(single(Ytr(:, idxStart:idxEnd)), 'CB');

            [loss, gradients] = dlfeval(@modelGradients, dlnet, xb, yb);

            [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
                trailingAvg, trailingAvgSq, iteration, learnRate);

            if mod(iteration, 50) == 0
                fprintf("Epoch %d/%d, iter %d, loss=%.6f\n", epoch, numEpochs, iteration, double(loss));
            end
        end

        % 简单 val
        if ~isempty(Xva)
            ypredVa = predict(dlnet, dlarray(single(Xva), 'CB'));
            valLoss = mean( (extractdata(ypredVa) - single(Yva)).^2, 'all' );
            fprintf("  >> valLoss=%.6f\n", valLoss);
        end
    end

    % ------------------------------------------------------------
    % 5. 保存网络
    % ------------------------------------------------------------
    matPath = fullfile(outDir, "deformer_mapper.mat");
    save(matPath, "dlnet");
    fprintf("Saved trained dlnetwork to %s\n", matPath);

    % ------------------------------------------------------------
    % 6. 导出 ONNX
    % ------------------------------------------------------------
    try
        lgraph = layerGraph(net);
        dlnetTmp = assembleNetwork(lgraph);
        onnxPath = fullfile(outDir, "deformer_mapper.onnx");
        exportONNXNetwork(dlnetTmp, onnxPath, ...
            'InputDataFormats','BC', ...
            'OutputDataFormats','BC');
        fprintf("Exported ONNX to %s\n", onnxPath);
    catch ME
        warning("ONNX 导出失败: %s", ME.message);
    end

    % ------------------------------------------------------------
    % 7. 写 prep.json （这里按你说的，不做归一化，就是 0/1）
    % ------------------------------------------------------------
    prep.muX = zeros(60,1);
    prep.sigX = ones(60,1);
    prep.muY = zeros(36,1);
    prep.sigY = ones(36,1);
    prep.flatOrder = char(flatOrder);

    prepPath = fullfile(outDir, "deformer_mapper_prep.json");
    fid = fopen(prepPath, 'w');
    fwrite(fid, jsonencode(prep, PrettyPrint=true), 'char');
    fclose(fid);
    fprintf("Wrote prep json to %s\n", prepPath);

    % ------------------------------------------------------------
    % 8. 写 selfcheck_bundle.npz
    %    选一条样本，把网络预测也一起写出去，给 Maya 做对比
    % ------------------------------------------------------------
    x_ref = single(X(:,1));
    y_ref = single(Y(:,1));
    y_pred = predict(dlnet, dlarray(x_ref, 'CB'));
    y_pred = extractdata(y_pred);

    bundle.x_ref = double(x_ref);
    bundle.y_ref = double(y_ref);
    bundle.y_pred_ref = double(y_pred);
    bundle.flatOrder = char(flatOrder);

    bundlePath = fullfile(outDir, "selfcheck_bundle.npz");
    write_npz(bundlePath, bundle);
    fprintf("Wrote selfcheck_bundle.npz to %s\n", bundlePath);

    fprintf("All done.\n");

end


% ================================================================
% 下面是用 Python 读单个 npz 的小函数
% ================================================================
function [x60, y36, flatOrder] = read_single_npz(fpath)
    % 用 Python 的 numpy 来读
    np = py.importlib.import_module('numpy');
    data = np.load(fpath, pyargs('allow_pickle', true));

    % 取 X_vec60 / Y_vec36
    if isfield(data, 'X_vec60')
        xv = data.get('X_vec60');
    else
        error("npz %s 没有 X_vec60", fpath);
    end
    if isfield(data, 'Y_vec36')
        yv = data.get('Y_vec36');
    else
        error("npz %s 没有 Y_vec36", fpath);
    end

    % 转成 MATLAB double
    x60 = double(xv.tolist());
    y36 = double(yv.tolist());

    % flatOrder
    if isfield(data, 'flatOrder')
        flatOrder = string(char(data.get('flatOrder').tolist()));
    else
        flatOrder = "row";
    end
end


% ================================================================
% 写 npz ：用 python 的 numpy.savez
% ================================================================
function write_npz(outPath, S)
    % S 是一个 struct
    np = py.importlib.import_module('numpy');

    % MATLAB -> Python array
    kwargs = pyargs();
    fns = fieldnames(S);
    for i = 1:numel(fns)
        key = fns{i};
        val = S.(key);
        % 转成 numpy array
        npVal = np.array(val);
        kwargs = [kwargs, key, npVal]; %#ok<AGROW>
    end

    np.savez(outPath, kwargs{:});
end


% ================================================================
% 简单 MLP 网络
% ================================================================
function lgraph = build_simple_mlp()

    layers = [
        featureInputLayer(60, Name="input", Normalization="none")
        fullyConnectedLayer(256, Name="fc1")
        reluLayer(Name="relu1")
        fullyConnectedLayer(256, Name="fc2")
        reluLayer(Name="relu2")
        fullyConnectedLayer(128, Name="fc3")
        reluLayer(Name="relu3")
        fullyConnectedLayer(36, Name="fc_out")
    ];

    lgraph = layerGraph(layers);
end


% ================================================================
% 损失 + 梯度
% ================================================================
function [loss, gradients] = modelGradients(dlnet, x, y)
    ypred = forward(dlnet, x);
    loss = mse(ypred, y);
    gradients = dlgradient(loss, dlnet.Learnables);
end
