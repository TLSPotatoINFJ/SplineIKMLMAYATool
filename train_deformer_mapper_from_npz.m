function train_deformer_mapper_from_npz(rootDir, outDir)
% 读取 Maya 单条样本 npz（X_mats/Y_mats/X_vec60/Y_vec36/flatOrder），
% 不归一化/SVD，训练 60->36 MLP，并导出对照 npz。
% 加强稳定性：更小学习率 + 余弦退火 + 梯度裁剪 + 样本一致性自检。
arguments
    rootDir (1,1) string
    outDir (1,1) string
end
%% ====== 配置 ======
cfg.forceFlatOrder = "column"; % 设为 "column" 可强制行展平；默认 "" 不强制
cfg.roundOutput = false; % 导出 npz 是否 round
cfg.roundDigits = 7;
cfg.consistencyThresh = 1e-3; % Y_mats 与 Y_vec36 最大绝对差阈值（超出报错）
nn.hidden = [1024 1024];
nn.numEpochs = 10000; % 够用即可，余弦退火会把 lr 收尾
nn.miniBatch = 64;
nn.baseLR = 5e-5; % ↓ 更小学习率（关键）
nn.minLR = 1e-7;
nn.scheduler = "cosine";
nn.stepIters = 2000; %#ok<NASGU> % 预留给 step 策略，不用
nn.stepGamma = 0.5; %#ok<NASGU>
nn.l2Factor = 0;
nn.dropoutRate = 0.0;
nn.earlyStopOn = false;
nn.clipNorm = 5.0; % 全局 L2 梯度裁剪
nn.tranWeight = 2.0; % 位移部分损失权重
%% ====== 基础检查 ======
if ~isfolder(rootDir), error("rootDir 不存在: %s", rootDir); end
if ~isfolder(outDir), mkdir(outDir); end
files = dir(fullfile(rootDir, "*.npz"));
if isempty(files), error("目录 %s 下没有 .npz", rootDir); end
fprintf("Found %d npz samples.\n", numel(files));
%% ====== 读入 + 一致性检查 ======
Xall = []; Yall = [];
flatOrder_all = "row";
sampleStructs = {};
badList = strings(0,1);
for i = 1:numel(files)
    fpath = fullfile(files(i).folder, files(i).name);
    try
        s = read_single_npz_maya_style(fpath); % struct: x60, y36, Xm, Ym, flatOrder
        % 可选：强制统一展平方式
        if ~isempty(cfg.forceFlatOrder)
            s.flatOrder = string(cfg.forceFlatOrder);
            s.x60 = pack_input_60_from_mats(s.Xm, s.flatOrder);
            s.y36 = pack_3mats_to_vec36_from_mats(s.Ym, s.flatOrder);
        end
        % 一致性自检：用 Ym 重新 pack 成 y36_check，看看与文件中的 y36 差多少
        y36_check = pack_3mats_to_vec36_from_mats(s.Ym, s.flatOrder);
        dmax = max(abs(y36_check(:) - s.y36(:)));
        if dmax > cfg.consistencyThresh
            badList(end+1) = string(files(i).name); %#ok<AGROW>
        end
    catch ME
        warning("跳过样本 %s: %s", fpath, ME.message);
        continue
    end
    if isempty(Xall)
        Xall = s.x60(:);
        Yall = s.y36(:);
        flatOrder_all = s.flatOrder;
    else
        Xall(:,end+1) = s.x60(:); %#ok<AGROW>
        Yall(:,end+1) = s.y36(:); %#ok<AGROW>
    end
    sampleStructs{end+1} = s; %#ok<AGROW>
end
if ~isempty(badList)
    fprintf("[WARN] 发现 %d 条样本的 Y_mats 与 Y_vec36 不一致（>|Δ|max>%.3g）\n", ...
        numel(badList), cfg.consistencyThresh);
    disp(badList);
    error("存在不一致的样本，请先修复该数据源。");
end
N = size(Xall,2);
if N==0, error("有效样本为 0，检查 .npz 格式。"); end
fprintf("Loaded %d valid samples.\n\n", N);
%% ====== 数据统计调试 ======
fprintf("[DEBUG] ---- 数据分布统计 ----\n");
tran_pos = [4,8,12,16,20,24,28,32,36];
trans_all = Yall(tran_pos, :);  % (9 x N)
trans1 = trans_all(1:3,:); trans2 = trans_all(4:6,:); trans3 = trans_all(7:9,:);
diff12 = mean(abs(trans1 - trans2), 'all');
diff13 = mean(abs(trans1 - trans3), 'all');
diff23 = mean(abs(trans2 - trans3), 'all');
fprintf("三个矩阵位移间平均|diff|: 1-2=%.4f, 1-3=%.4f, 2-3=%.4f\n", diff12, diff13, diff23);
fprintf("位移整体 mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", mean(trans_all(:)), std(trans_all(:)), min(trans_all(:)), max(trans_all(:)));
rot_pos = setdiff(1:36, tran_pos);
rot_all = Yall(rot_pos, :);
fprintf("旋转整体 mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n", mean(rot_all(:)), std(rot_all(:)), min(rot_all(:)), max(rot_all(:)));
fprintf("flatOrder_all: %s\n", flatOrder_all);
fprintf("[DEBUG] ------------------------\n\n");
%% ====== 训练前：打印一条未编辑原样本的完整 y_vec36 ======
idx_for_print = pick_sample_by_yvec_has_translation(Yall);
s_print = sampleStructs{idx_for_print};
fprintf("[DEBUG] ---- 原始样本 #%d 的完整 y_vec36（未做编辑）----\n", idx_for_print);
disp(s_print.y36(:));
fprintf("[DEBUG] -------------------------------------------------\n\n");
%% ====== 划分（稳健） ======
perm = randperm(N);
if N == 1
    valIdx = 1; trnIdx = [];
else
    valRatio = 0.1;
    nVal = max(1, min(round(N*valRatio), N-1));
    valIdx = perm(1:nVal);
    trnIdx = perm(nVal+1:end);
end
Xtr = single(Xall(:,trnIdx)); Ytr = single(Yall(:,trnIdx));
Xv = single(Xall(:,valIdx)); Yv = single(Yall(:,valIdx));
% ====== [FIX] N=1 时强制用全数据训练（无验证集） ======
if isempty(trnIdx)
    trnIdx = valIdx;
    valIdx = [];
    Xtr = single(Xall(:,trnIdx));
    Ytr = single(Yall(:,trnIdx));
    Xv = []; Yv = [];
    fprintf("[FIX] N=1 时强制用全数据训练（无验证集）。\n");
end
%% ====== GPU ======
useGPU = canUseGPU;
if useGPU
    gpuDevice([]); fprintf("🟢 Using GPU for training.\n");
else
    fprintf("⚪ GPU not available, using CPU.\n");
end
%% ====== 网络 ======
layers = [
    featureInputLayer(60,"Name","in","Normalization","none")
];
for i = 1:numel(nn.hidden)
    h = nn.hidden(i);
    layers = [layers; fullyConnectedLayer(h,"Name","fc"+i); reluLayer("Name","relu"+i)];
    if nn.dropoutRate>0
        layers = [layers; dropoutLayer(nn.dropoutRate,"Name","drop"+i)]; %#ok<AGROW>
    end
end
layers = [layers; fullyConnectedLayer(36,"Name","fc_out")];
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
if useGPU
    net = dlupdate(@gpuArray, net);
    if ~isempty(Xv)
        Xv = gpuArray(Xv);
        Yv = gpuArray(Yv);
    end
end
%% ====== 初始损失检查 ======
fprintf("\n[DEBUG] Computing initial loss...\n");
if ~isempty(Xtr)
    dlX = dlarray(Xtr,"CB");
    dlY = dlarray(Ytr,"CB");
else
    dlX = dlarray(Xv,"CB");
    dlY = dlarray(Yv,"CB");
end
Yinit = forward(net, dlX);
initLoss = mse(Yinit, dlY);
fprintf("Initial MSE loss: %f\n\n", gather(double(extractdata(initLoss))));
%% ====== 训练进度图 ======
fig = figure('Name','Training Progress','NumberTitle','off');
ax = axes(fig); hold(ax,'on'); grid(ax,'on');
yyaxis left;
hTrain = animatedline('LineWidth',1.6,'DisplayName','Train MSE');
hVal = animatedline('LineWidth',1.6,'DisplayName','Val MSE');
ylabel('Loss (MSE)');
yyaxis right;
hLR = animatedline('LineWidth',1.2,'DisplayName','LR');
ylabel('Learning Rate');
xlabel('Iteration');
legend('show','Location','northeast');
title('Train / Val Loss & LR');
drawnow;
%% ====== 训练（带梯度裁剪） ======
beta1=0.9; beta2=0.999; epsilon=1e-8;
avgGrad=[]; avgSqGrad=[];
itersPerEpoch = max(1, ceil(max(1,size(Xtr,2))/max(1,nn.miniBatch)));
totalItersEst = max(1, nn.numEpochs * itersPerEpoch);
iter=0; logI=[]; logT=[]; logV=[]; logLR=[];
for e = 1:nn.numEpochs
    if ~isempty(trnIdx)
        ord = randperm(size(Xtr,2));
        Xtr = Xtr(:,ord); Ytr = Ytr(:,ord);
    end
    for t = 1:max(1,nn.miniBatch):max(1,size(Xtr,2))
        if isempty(trnIdx)
            Xb = Xv; Yb = Yv; % 仅为可视化连贯
        else
            sel = t:min(t+nn.miniBatch-1, size(Xtr,2));
            Xb = Xtr(:,sel); Yb=Ytr(:,sel);
        end
        if useGPU, Xb=gpuArray(Xb); Yb=gpuArray(Yb); end
        lr = scheduleLR(nn, iter+1, totalItersEst);
        [grad,loss] = dlfeval(@modelGradientsL2, net, dlarray(Xb,"CB"), dlarray(Yb,"CB"), nn.l2Factor, nn.tranWeight);
        % ---- 全局 L2 梯度裁剪 ----
        gn = globalGradL2Norm(grad);
        if gn > nn.clipNorm
            scale = nn.clipNorm / (gn + 1e-12);
            grad = dlupdate(@(g) g*scale, grad);
        end
        [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad,iter+1,lr,beta1,beta2,epsilon);
        iter = iter + 1;
        % 验证
        if ~isempty(Xv)
            Yv_pred = forward(net, dlarray(Xv,"CB"));
            valLoss = mse(Yv_pred, dlarray(Yv,"CB"));
            valLoss = gather(double(extractdata(valLoss)));
        else
            valLoss = gather(double(extractdata(loss)));  % 用 train loss 代替
        end
        trainLoss = gather(double(extractdata(loss)));
        yyaxis left; addpoints(hTrain, iter, trainLoss); addpoints(hVal, iter, valLoss);
        yyaxis right; addpoints(hLR, iter, lr);
        drawnow limitrate;
        logI(end+1)=iter; logT(end+1)=trainLoss; logV(end+1)=valLoss; logLR(end+1)=lr; %#ok<AGROW>
        if mod(iter,200)==0
            fprintf(" iter=%d lr=%g train=%f val=%f (gradNorm=%.3f)\n", iter, lr, trainLoss, valLoss, gn);
        end
    end
end
fprintf("Training done.\n\n");
%% ====== 训练后：打印训练样本 True vs Pred ======
net_cpu = dlupdate(@gather, net);
train_id = choose_random_index(trnIdx, N);
x_tr_dbg = Xall(:, train_id);
y_tr_true = Yall(:, train_id);
y_tr_pred = predict_raw(net_cpu, single(x_tr_dbg));
diff_abs = abs(y_tr_pred(:) - y_tr_true(:));
fprintf("[DEBUG] ---- 训练样本 #%d: true vs pred ----\n", train_id);
fprintf(" max(|diff|) = %g\n", max(diff_abs));
fprintf(" mean(|diff|)= %g\n", mean(diff_abs));
fprintf(" y_tr_true(1:36) = \n"); disp(y_tr_true(:)');
fprintf(" y_tr_pred(1:36) = \n"); disp(y_tr_pred(:)');
% ====== 额外调试：位移和旋转单独指标 ======
tran_pos = [4,8,12,16,20,24,28,32,36];
y_true_tran = y_tr_true(tran_pos);
y_pred_tran = y_tr_pred(tran_pos);
fprintf("True 位移 (tx1,ty1,tz1, tx2,ty2,tz2, tx3,ty3,tz3): \n"); disp(y_true_tran(:)');
fprintf("Pred 位移: \n"); disp(y_pred_tran(:)');
diff_tran = abs(y_true_tran - y_pred_tran);
fprintf("位移 |diff| max=%.4f, mean=%.4f\n", max(diff_tran), mean(diff_tran));
rot_pos = setdiff(1:36, tran_pos);
mean_rot_diff = mean(abs(y_tr_true(rot_pos) - y_tr_pred(rot_pos)));
fprintf("旋转 mean(|diff|)= %.4f\n", mean_rot_diff);
fprintf("[DEBUG] -------------------------------------------\n\n");
%% ====== 自检并导出 ======
sampleIdx = randi(N);
s0 = sampleStructs{sampleIdx};
x_te = Xall(:,sampleIdx); y_te = Yall(:,sampleIdx);
y_pd = predict_raw(net_cpu, single(x_te));
rmse = sqrt(mean((y_pd(:) - y_te(:)).^2));
fprintf("Final self-check RMSE on 1 sample: %f\n", rmse);
rmse_tran = sqrt(mean((y_pd(tran_pos) - y_te(tran_pos)).^2));
fprintf("Self-check RMSE translation: %f\n", rmse_tran);
rmse_rot = sqrt(mean((y_pd(rot_pos) - y_te(rot_pos)).^2));
fprintf("Self-check RMSE rotation: %f\n", rmse_rot);
selfInputPath = fullfile(outDir, "selfcheck_input.npz");
save_npz_full(selfInputPath, s0.Xm, s0.Ym, s0.x60, s0.y36, s0.flatOrder, cfg);
selfPredPath = fullfile(outDir, "selfcheck_pred.npz");
Ym_pred = vec36_to_3mats(y_pd(:), s0.flatOrder);
save_npz_full(selfPredPath, s0.Xm, Ym_pred, s0.x60, y_pd(:), s0.flatOrder, cfg);
fprintf("Self-check npz exported to:\n %s\n %s\n\n", selfInputPath, selfPredPath);
%% ====== 训练后额外调试：多样本 true vs pred + 最终统计 ======
fprintf("[DEBUG] ---- 训练后额外调试信息 ----\n");
fprintf("最终 train loss: %.6f\n", logT(end));
fprintf("最终 val loss: %.6f\n", logV(end));
% 多样本检查（最多5个）
nCheck = min(5, N);
for k = 1:nCheck
    sid = randi(N);
    y_true_k = Yall(:,sid);
    y_pred_k = predict_raw(net_cpu, single(Xall(:,sid)));
    rmse_k = sqrt(mean((y_pred_k(:) - y_true_k(:)).^2));
    rmse_tran_k = sqrt(mean((y_pred_k(tran_pos) - y_true_k(tran_pos)).^2));
    rmse_rot_k = sqrt(mean((y_pred_k(rot_pos) - y_true_k(rot_pos)).^2));
    fprintf("样本 #%d: RMSE total=%.6f, tran=%.6f, rot=%.6f\n", sid, rmse_k, rmse_tran_k, rmse_rot_k);
    fprintf("  Pred 位移: "); disp(y_pred_k(tran_pos(:))');
end
% 整体预测统计
Ypred_all = zeros(36, N);
for k = 1:N
    Ypred_all(:,k) = predict_raw(net_cpu, single(Xall(:,k)));
end
trans_pred = Ypred_all(tran_pos, :);
trans1_p = trans_pred(1:3,:); trans2_p = trans_pred(4:6,:); trans3_p = trans_pred(7:9,:);
diff12_p = mean(abs(trans1_p - trans2_p), 'all');
diff13_p = mean(abs(trans1_p - trans3_p), 'all');
diff23_p = mean(abs(trans2_p - trans3_p), 'all');
fprintf("预测位移间平均|diff|: 1-2=%.4f, 1-3=%.4f, 2-3=%.4f\n", diff12_p, diff13_p, diff23_p);
fprintf("预测位移整体 mean=%.4f, std=%.4f\n", mean(trans_pred(:)), std(trans_pred(:)));
fprintf("[DEBUG] -----------------------------\n\n");
%% ====== 保存模型 & ONNX ======
prep.muX = zeros(60,1); prep.sigX = ones(60,1);
prep.muY = zeros(36,1); prep.sigY = ones(36,1);
prep.flatOrder = char(flatOrder_all);
fid = fopen(fullfile(outDir,"deformer_mapper_prep.json"),'w');
fwrite(fid, jsonencode(prep)); fclose(fid);
save(fullfile(outDir,"deformer_mapper_dlnet.mat"),"net_cpu","-v7.3");
onnxPath = fullfile(outDir,"deformer_mapper.onnx");
try
    if exist('exportONNXNetwork','file')==2
        lgraph2 = assignLearnables(layerGraph(layers), net_cpu);
        exportONNXNetwork(lgraph2, onnxPath, "OpsetVersion", 17);
        fprintf("ONNX exported: %s\n", onnxPath);
    else
        fprintf("exportONNXNetwork 不存在，跳过 ONNX 导出。\n");
    end
catch ME
    warning("导出 ONNX 失败：%s", ME.message);
end
%% ====== 曲线 & CSV ======
saveas(fig, fullfile(outDir,'combined_loss_curve.png'));
T=table(logI(:),logT(:),logV(:),logLR(:), ...
    'VariableNames',{'iteration','train_loss','val_loss','learning_rate'});
writetable(T, fullfile(outDir,'training_log.csv'));
fprintf("✅ All done! Output in %s\n", outDir);
fprintf(" Self-check RMSE: %f\n", rmse);
end % ===== 主函数 =====
%% ====== 工具函数 ======
function idx = choose_random_index(trnIdx, N)
if isempty(trnIdx), idx = randi(N); else, idx = trnIdx(randi(numel(trnIdx))); end
end
function idx = pick_sample_by_yvec_has_translation(Yall)
idx = 1;
if isempty(Yall), return; end
tran_idx = [4 8 12 16 20 24 28 32 36];
for i = 1:size(Yall,2)
    y = Yall(:,i);
    if any(abs(y(tran_idx)) > 1e-8), idx = i; return; end
end
idx = randi(size(Yall,2));
end
function s = read_single_npz_maya_style(fpath)
np = py.importlib.import_module('numpy');
data = np.load(fpath, pyargs("allow_pickle",true));
if ~isempty(data.get('flatOrder'))
    try, flatOrder = string(char(data.get('flatOrder').tolist()));
    catch, flatOrder = string(char(data.get('flatOrder')));
    end
else
    flatOrder = "row";
end
Xm = []; Ym = [];
if ~isempty(data.get('X_mats')), Xm = double(py.numpy.array(data.get('X_mats'))); end
if ~isempty(data.get('Y_mats')), Ym = double(py.numpy.array(data.get('Y_mats'))); end
hasXvec = ~isempty(data.get('X_vec60'));
hasYvec = ~isempty(data.get('Y_vec36'));
if hasXvec
    xv = data.get('X_vec60');
    x60 = double(py.array.array('d', py.numpy.nditer(xv))).'; x60 = x60(:);
else
    x60 = pack_input_60_from_mats(Xm, flatOrder);
end
if hasYvec
    yv = data.get('Y_vec36');
    y36 = double(py.array.array('d', py.numpy.nditer(yv))).'; y36 = y36(:);
else
    y36 = pack_3mats_to_vec36_from_mats(Ym, flatOrder);
end
if isempty(Xm), Xm = vec60_to_5mats(x60, flatOrder); end
if isempty(Ym), Ym = vec36_to_3mats(y36, flatOrder); end
s = struct('x60',x60,'y36',y36,'Xm',Xm,'Ym',Ym,'flatOrder',flatOrder);
end
function v60 = pack_input_60_from_mats(Xm, flatOrder)
v60 = zeros(60,1);
for i = 1:5
    M34 = Xm(1:3,1:4,i);
    if flatOrder=="row", v = reshape(M34,12,1); else, v = reshape(M34.',12,1); end
    v60((i-1)*12+1:i*12) = v;
end
end
function v36 = pack_3mats_to_vec36_from_mats(Ym, flatOrder)
v36 = zeros(36,1);
for i = 1:3
    M34 = Ym(1:3,1:4,i);
    if flatOrder=="row", v = reshape(M34,12,1); else, v = reshape(M34.',12,1); end
    v36((i-1)*12+1:i*12) = v;
end
end
function Xm = vec60_to_5mats(v60, flatOrder)
v60 = v60(:); Xm = zeros(4,4,5);
for i = 1:5
    seg = v60((i-1)*12+1:i*12);
    if flatOrder=="row", M34 = reshape(seg,[3,4]); else, M34 = reshape(seg,[4,3]).'; end
    M = eye(4); M(1:3,1:4) = M34; Xm(:,:,i) = M;
end
end
function Ym = vec36_to_3mats(v36, flatOrder)
v36 = v36(:); Ym = zeros(4,4,3);
for i = 1:3
    seg = v36((i-1)*12+1:i*12);
    if flatOrder=="row", M34 = reshape(seg,[3,4]); else, M34 = reshape(seg,[4,3]).'; end
    M = eye(4); M(1:3,1:4) = M34; Ym(:,:,i) = M;
end
end
function save_npz_full(fpath, Xm, Ym, x60, y36, flatOrder, cfg)
np = py.importlib.import_module('numpy');
if cfg.roundOutput
    Xm = round(Xm, cfg.roundDigits);
    Ym = round(Ym, cfg.roundDigits);
    x60 = round(x60, cfg.roundDigits);
    y36 = round(y36, cfg.roundDigits);
end
Xm_py = np.array(single(Xm), pyargs('order','F')); % (4,4,5)
Ym_py = np.array(single(Ym), pyargs('order','F')); % (4,4,3)
x60_py = np.array(single(reshape(x60,[60 1])));
y36_py = np.array(single(reshape(y36,[36 1])));
fo_py = py.str(flatOrder);
np.savez(fpath, pyargs( ...
    'X_mats',Xm_py,'Y_mats',Ym_py, ...
    'X_vec60',x60_py,'Y_vec36',y36_py, ...
    'flatOrder',fo_py));
end
function lr = scheduleLR(cfg, iter, totalItersEst)
switch cfg.scheduler
    case "cosine"
        T = max(1,totalItersEst);
        lr = cfg.minLR + 0.5*(cfg.baseLR - cfg.minLR)*(1 + cos(pi*(iter-1)/T));
    case "step"
        steps = floor((iter-1)/max(1,cfg.stepIters));
        lr = cfg.baseLR * (cfg.stepGamma ^ steps);
    otherwise
        lr = cfg.baseLR;
end
end
function [gradients, loss] = modelGradientsL2(net, Xb, Yb, l2Factor, tranWeight)
Yhat = forward(net, Xb);
diff = Yhat - Yb;
tran_pos = [4,8,12,16,20,24,28,32,36];
weights = ones(size(Yb,1),1); weights(tran_pos) = tranWeight;
mseLoss = mean(weights .* (diff.^2), 'all');
reg = 0;
P = net.Learnables;
for i = 1:size(P,1)
    if endsWith(string(P.Parameter(i)),"Weights")
        reg = reg + sum(P.Value{i}.^2,'all');
    end
end
loss = mseLoss + l2Factor*reg;
gradients = dlgradient(loss, net.Learnables);
end
function n = globalGradL2Norm(grad)
% grad 是 Learnables 表的 Value 单元格构成的 cell/table；计算全局 L2 范数
n = 0;
vals = grad.Value;
for i = 1:numel(vals)
    g = vals{i};
    if ~isempty(g)
        n = n + sum(g.^2,'all');
    end
end
n = sqrt(gather(double(n)));
end
function y_pred = predict_raw(net, x)
dlx = dlarray(single(x),"CB");
dly = forward(net, dlx);
y_pred = double(extractdata(dly));
end
function lgraphOut = assignLearnables(lgraphIn, dlnet)
lgraphOut = lgraphIn;
params = dlnet.Learnables;
for i = 1:size(params,1)
    layerName = params.Layer(i);
    paramName = params.Parameter(i);
    val = gather(extractdata(params.Value{i}));
    hit = strcmp({lgraphOut.Layers.Name}, layerName);
    if any(hit)
        L = lgraphOut.Layers(hit);
        if isprop(L, paramName)
            L.(paramName) = val;
            lgraphOut = replaceLayer(lgraphOut, layerName, L);
        end
    end
end
end