function train_deformer_mapper_assemble(rootDir, prefix, outDir)
%TRAIN_DEFORMER_MAPPER_ASSEMBLE
% 修复版：避免 regressionLayer 拼接问题
arguments
    rootDir (1,1) string
    prefix (1,1) string
    outDir (1,1) string
end
if ~isfolder(outDir), mkdir(outDir); end

%% ---------- 1. 读取 manifest ----------
manifestPath = fullfile(rootDir, prefix + "_manifest.json");
if ~isfile(manifestPath)
    error("未找到 manifest.json");
end
man = jsondecode(fileread(manifestPath));
flatOrder = string(man.flatOrder);
shards = arrayfun(@(x) fullfile(rootDir, x.file), man.shards, "UniformOutput", false);
fprintf("找到 %d 个数据分片\n", numel(shards));

%% ---------- 2. 计算 X 归一化 ----------
fprintf("计算 X 的 mean/std...\n");
muX = zeros(60,1); M2 = zeros(60,1); N = 0;
X_all = [];
for s = 1:numel(shards)
    data = py.numpy.load(shards{s}, pyargs("allow_pickle",true, "mmap_mode","r"));
    Xpy = data.get('X_vec60');
    X = double(py.numpy.array(Xpy)); K = size(X,2); X = reshape(X,60,K);
    X_all = [X_all, X];
    for k = 1:K
        d = X(:,k) - muX;
        N = N + 1;
        muX = muX + d/N;
        M2 = M2 + d.*(X(:,k)-muX);
    end
end
sigX = sqrt(M2 / max(1,N-1));
fprintf("X 归一化完成 (N=%d)\n", N);

%% ---------- 3. 训练 16 个小模型 ----------
nets = cell(16,1);
channelMap = struct();
ch = 1;
for mat = 1:3
    for row = 1:4
        for col = 1:4
            if mat == 3 && row == 4 && col == 4, continue; end
            channelMap.(sprintf('ch%02d', ch-1)) = struct('mat',mat,'row',row,'col',col);
            fprintf("训练通道 %02d: [%d,%d,%d]\n", ch-1, mat, row, col);
            
            % 提取 Y 通道
            Y_ch = [];
            for s = 1:numel(shards)
                data = py.numpy.load(shards{s}, pyargs("allow_pickle",true, "mmap_mode","r"));
                Ypy = data.get('Y_vec36');
                Y = double(py.numpy.array(Ypy)); K = size(Y,2);
                idx = (mat-1)*12 + (row-1)*4 + col;
                Y_ch = [Y_ch, Y(idx,:)];
            end
            Y_ch = Y_ch(:)';
            
            % 小网络（不含 regressionLayer）
            layers = [
                featureInputLayer(60,"Normalization","none")
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(1)
            ];
            net = dlnetwork(layerGraph(layers));
            
            % 手动训练（MSE 损失）
            numEpochs = 300; miniBatch = 2048; lr = 2e-3;
            X_norm = (X_all - muX) ./ sigX;
            velocity = [];
            for epoch = 1:numEpochs
                idx = randperm(size(X_norm,2));
                for i = 1:miniBatch:size(X_norm,2)
                    sel = idx(i:min(i+miniBatch-1, end));
                    Xb = dlarray(single(X_norm(:,sel)),"CB");
                    Yb = dlarray(single(Y_ch(sel)),"CB");
                    [grad, loss] = dlfeval(@modelGradients, net, Xb, Yb);
                    [net, velocity] = sgdmupdate(net, grad, velocity, lr);
                end
                if mod(epoch, 10) == 0
                    fprintf("  Epoch %d, Loss = %.6f\n", epoch, double(gather(extractdata(loss))));
                end
            end
            nets{ch} = net;
            ch = ch + 1;
        end
    end
end

%% ---------- 4. 组装大模型 ----------
sharedLayers = [
    featureInputLayer(60,"Normalization","none")
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
];
lgraph = layerGraph(sharedLayers);

% 添加 16 个输出头
for i = 1:16
    head = fullyConnectedLayer(1);
    lgraph = addLayers(lgraph, head);
    lgraph = connectLayers(lgraph, 'relu2', sprintf('fc%d', i));
end

% 复制权重
bigNet = dlnetwork(lgraph);
for i = 1:16
    small = nets{i};
    for j = 1:4
        bigNet.Learnables.Value{j} = small.Learnables.Value{j};
    end
    bigNet.Learnables.Value{4+i} = small.Learnables.Value{5};
end

%% ---------- 5. 导出 ----------
prep.muX = muX; prep.sigX = sigX; prep.flatOrder = char(flatOrder);
jsonPath = fullfile(outDir, "deformer_mapper_prep.json");
fid = fopen(jsonPath,'w'); fwrite(fid,jsonencode(prep)); fclose(fid);

onnxPath = fullfile(outDir, "deformer_mapper.onnx");
exportONNXNetwork(bigNet, onnxPath);
fprintf("模型导出完成：%s\n", onnxPath);

%% ---------- 6. selfcheck ----------
data_ref = py.numpy.load(shards{1}, pyargs("allow_pickle",true));
X_ref = double(py.numpy.array(data_ref.get('X_vec60')));
x_ref = X_ref(:,1); xn = (x_ref - muX) ./ sigX;

y_pred_36 = zeros(36,1);
ch = 1;
for mat = 1:3
    for row = 1:4
        for col = 1:4
            if mat == 3 && row == 4 && col == 4, continue; end
            y = forward(nets{ch}, dlarray(single(xn),"CB"));
            y_pred_36((mat-1)*12 + (row-1)*4 + col) = double(gather(extractdata(y)));
            ch = ch + 1;
        end
    end
end

y_pred_36 = project_pred_svd(dlarray(single(y_pred_36),"CB"), flatOrder);
y_pred_36 = double(extractdata(y_pred_36));

x_ref_py = py.numpy.array(x_ref(:)');
y_pred_py = py.numpy.array(y_pred_36(:)');

py.numpy.savez(fullfile(outDir,'selfcheck_bundle.npz'), ...
    pyargs('x_ref', x_ref_py, 'y_pred_ref', y_pred_py, ...
           'muX', py.numpy.array(muX(:)'), 'sigX', py.numpy.array(sigX(:)'), ...
           'flatOrder', char(flatOrder)));

fprintf("训练完成！\n");
end

% ====== 梯度函数 ======
function [gradients, loss] = modelGradients(net, X, Y)
Ypred = forward(net, X);
loss = mse(Ypred, Y);
gradients = dlgradient(loss, net.Learnables);
end

% ====== SVD 投影 ======
function Y = project_pred_svd(Yhat, flatOrder)
Yn = double(extractdata(Yhat)); B = size(Yn,2);
Yblk = reshape(Yn,12,3,B);
for blk = 1:3
    seg = Yblk(1:12,blk,:);
    if flatOrder == "row"
        Rb = reshape(seg(1:9,:),3,3,B);
        t = reshape(seg(10:12,:),3,1,B);
    else
        Rb = reshape(seg(1:9,:),3,3,B)';
        t = reshape(seg(10:12,:),3,1,B);
    end
    for i = 1:B
        M = Rb(:,:,i);
        [U,~,V] = svd(M,'econ'); R = U*V'; if det(R)<0, U(:,3)=-U(:,3); R=U*V'; end
        Rb(:,:,i) = R;
    end
    if flatOrder == "row"
        Yblk(1:9,blk,:) = reshape(Rb,9,1,B);
    else
        Yblk(1:9,blk,:) = reshape(Rb',9,1,B);
    end
    Yblk(10:12,blk,:) = reshape(t,3,1,B);
end
Y = dlarray(reshape(Yblk,36,B),"CB");
end