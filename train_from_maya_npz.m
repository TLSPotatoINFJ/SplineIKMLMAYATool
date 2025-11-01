function train_from_maya_npz(datasetDir, onnxOut, jsonOut, matOut)
% 从 Maya 导出的 .npz 样本训练全连接回归网络（60->36）
% 并导出：ONNX 模型 + 预处理 JSON + 备份 MAT
%
% 用法：
% train_from_maya_npz("D:\WKS\Samples", ...
%                     "D:\WKS\deformer_mapper.onnx", ...
%                     "D:\WKS\deformer_mapper_prep.json", ...
%                     "D:\WKS\deformer_mapper_trained.mat");

arguments
    datasetDir (1,1) string
    onnxOut    (1,1) string
    jsonOut    (1,1) string
    matOut     (1,1) string
end

%% 0) 读取数据（兼容 X_vec60/Y_vec36 或 X_mats/Y_mats）
[X, Y, meta] = i_read_all_npz(datasetDir);   % X:60×N, Y:36×N

%% 1) 标准化（与推理端一致）
muX  = mean(X, 2);
sigX = std(X, 0, 2);
sigX = max(sigX, 1e-8);          % 避免除0
Xn   = (X - muX) ./ sigX;

%% 2) 网络结构（非线性回归 MLP，可按需调参）
layers = [
    featureInputLayer(60, Normalization="none", Name="in")
    fullyConnectedLayer(256, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(256, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(128, Name="fc3")
    reluLayer(Name="relu3")
    fullyConnectedLayer(36, Name="out")
];

% 需要 R2023b+ 提供 trainnet；更老版本我可以给你写自定义训练循环
assert(exist("trainnet","file")==2, ...
    "未检测到 trainnet（建议 MATLAB R2023b+）。如果你用老版，我可以给你提供自定义训练循环版本。");

% 训练数据 -> dlarray（CB：C=特征通道，B=批）
dlX = dlarray(single(Xn), "CB");  % 60×N
dlY = dlarray(single(Y ), "CB");  % 36×N

% 训练选项（可调）
opts = trainingOptions("adam", ...
    MaxEpochs=200, ...
    InitialLearnRate=5e-3, ...
    MiniBatchSize=256, ...
    Shuffle="every-epoch", ...
    L2Regularization=1e-5, ...
    Plots="none", ...
    Verbose=true);

% 包装损失
lgraph = layerGraph(layers);
dlnet  = dlnetwork(lgraph);
modelFcn = @(net, Xb, Yb) i_modelLoss(net, Xb, Yb);  % 返回 [loss, state, Ypred]

% 开训
dlnet = trainnet(dlX, dlY, dlnet, modelFcn, opts);

%% 3) 导出 ONNX + 预处理 JSON + 备份 MAT
% ONNX（需要 Deep Learning Toolbox Converter for ONNX）
exportONNXNetwork(dlnet, onnxOut, "OpsetVersion", 17);
fprintf("✅ 导出 ONNX: %s\n", onnxOut);

prep.muX      = muX;
prep.sigX     = sigX;
prep.flatOrder= meta.flatOrder;   % Maya 端默认为 'row'
jsonText = jsonencode(prep);
fid = fopen(jsonOut,'w'); assert(fid>0, "无法写入JSON：%s", jsonOut);
fwrite(fid, jsonText); fclose(fid);
fprintf("✅ 导出 预处理 JSON: %s\n", jsonOut);

save(matOut, "dlnet", "muX", "sigX", "meta");
fprintf("✅ 备份 MAT: %s\n", matOut);

end % === train_from_maya_npz ===


%% ===== 损失封装（MSE） =====
function [loss, state, Ypred] = i_modelLoss(net, Xb, Yb)
    [Ypred, state] = forward(net, Xb);
    loss = mse(Ypred, Yb);
end


%% ===== 读取 .npz 数据并拼训练集 =====
function [X, Y, meta] = i_read_all_npz(datasetDir)
% 扫描目录所有 .npz，读取并规范为 X:60×N, Y:36×N
S = dir(fullfile(datasetDir, "*.npz"));
assert(~isempty(S), "目录中未找到 *.npz：%s", datasetDir);

% 准备 Python/numpy
pe = pyenv;
if pe.Status=="NotLoaded"
    % 如需要，可在此指定你的 Python:
    % pyenv("Version","C:\Path\To\Python39\python.exe");
end
try
    ~py.importlib.import_module('numpy');
catch ME
    error("MATLAB 无法导入 numpy。请在该 Python 安装：pip install numpy\n%s", ME.message);
end

X = []; Y = [];
flatOrders = strings(0);
filesKept  = strings(0);

for k = 1:numel(S)
    f = fullfile(S(k).folder, S(k).name);
    npz = py.numpy.load(f, pyargs("allow_pickle", true));
    keys = string(cell(npz.files.tolist()));
    flatOrder = "row";
    if any(keys=="flatOrder")
        try
            flatOrder = string(npz.get("flatOrder").tolist());
        catch
            flatOrder = string(npz.get("flatOrder"));
        end
        if ~(flatOrder=="row" || flatOrder=="col"), flatOrder="row"; end
    end

    % 取 X
    if any(keys=="X_vec60")
        x_vec = double(cell(npz.get("X_vec60").tolist()));
        x_vec = x_vec(:);
        if numel(x_vec) ~= 60, warning("跳过X_vec60!=60：%s", f); continue; end
    elseif any(keys=="X_mats")
        x_mats = i_np_to_mat(npz.get("X_mats"));    % 4x4x5
        if ~isequal(size(x_mats), [4 4 5]), warning("跳过X_mats尺寸异常：%s", f); continue; end
        x_vec = i_mats5_to_vec60(x_mats, flatOrder);
    else
        warning("跳过（无X）：%s", f); continue;
    end

    % 取 Y
    if any(keys=="Y_vec36")
        y_vec = double(cell(npz.get("Y_vec36").tolist()));
        y_vec = y_vec(:);
        if numel(y_vec) ~= 36, warning("跳过Y_vec36!=36：%s", f); continue; end
    elseif any(keys=="Y_mats")
        y_mats = i_np_to_mat(npz.get("Y_mats"));    % 4x4x3
        if ~isequal(size(y_mats), [4 4 3]), warning("跳过Y_mats尺寸异常：%s", f); continue; end
        y_vec = i_mats3_to_vec36(y_mats, flatOrder);
    else
        warning("跳过（无Y）：%s", f); continue;
    end

    X(:,end+1) = x_vec; %#ok<AGROW>
    Y(:,end+1) = y_vec; %#ok<AGROW>
    flatOrders(end+1) = flatOrder; %#ok<AGROW>
    filesKept(end+1)  = string(f); %#ok<AGROW>
end

assert(~isempty(X), "没有有效样本被读取：%s", datasetDir);

meta = struct();
meta.files     = filesKept;
% 如果不同样本 flatOrder 不一致，取出现最多的
meta.flatOrder = mode(flatOrders);

end


%% ===== Python ndarray -> MATLAB 数组 =====
function M = i_np_to_mat(np_array)
    c = np_array.tolist();
    C = cell(c);
    M = i_cell2mat_recursive(C);
end

function out = i_cell2mat_recursive(C)
    if ~iscell(C), out = C; return; end
    if isempty(C), out = []; return; end
    if iscell(C{1})
        out = cellfun(@i_cell2mat_recursive, C, 'UniformOutput', false);
        out = cell2mat(out);
    else
        out = cellfun(@double, C);
    end
end


%% ===== 摊平工具（与 Maya 端一致） =====
function x = i_mats5_to_vec60(Xmats, flatOrder)
% Xmats: 4x4x5  ->  60x1
x = zeros(60,1);
for i = 1:5
    M = Xmats(:,:,i);
    M34 = M(1:3,1:4);
    if flatOrder=="row"
        v = reshape(M34, [], 1);
    else
        v = reshape(M34.', [], 1);
    end
    x( (i-1)*12+(1:12) ) = v;
end
end

function y = i_mats3_to_vec36(Ymats, flatOrder)
% Ymats: 4x4x3  ->  36x1
y = zeros(36,1);
for i = 1:3
    M = Ymats(:,:,i);
    M34 = M(1:3,1:4);
    if flatOrder=="row"
        v = reshape(M34, [], 1);
    else
        v = reshape(M34.', [], 1);
    end
    y( (i-1)*12+(1:12) ) = v;
end
end
