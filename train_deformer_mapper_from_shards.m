function train_deformer_mapper_from_shards(rootDir, prefix, outDir)
% ==========================================================
% TLS Deformer Mapper Trainer (Train + Validation curves + Best Checkpoint)
% - 读取 shard(.npz) + manifest
% - 实时绘制 Train/Val Loss 双曲线
% - 标注并保存"最佳验证损失"模型 checkpoint_best.mat
% - 导出 prep.json / dlnet.mat / 可选 ONNX
% ==========================================================
arguments
    rootDir (1,1) string
    prefix  (1,1) string
    outDir  (1,1) string
end

if ~isfolder(outDir), mkdir(outDir); end

%% ---------- 读取或生成 manifest ----------
manifestPath = fullfile(rootDir, prefix + "_manifest.json");
if ~isfile(manifestPath)
    fprintf("Manifest not found, scanning '%s'...\n", rootDir);
    files = dir(fullfile(rootDir, prefix + "_*.npz"));
    assert(~isempty(files), "No shards found with prefix '%s' in %s", prefix, rootDir);

    shardsList = struct('file', {}, 'num_samples', {}, 'start_idx', {});
    total = 0; flatOrder = "row"; store_vec = true;
    for i = 1:numel(files)
        fpath = fullfile(files(i).folder, files(i).name);
        data  = py.numpy.load(fpath, pyargs("allow_pickle",true));
        if ~isempty(data.get('X_vec60'))
            Xpy = data.get('X_vec60'); sh = cell(Xpy.shape); K = double(sh{2});
            store_vec = true;
        else
            XmPy = data.get('X_mats'); sh = cell(XmPy.shape); K = double(sh{4});
            store_vec = false;
        end
        if ~isempty(data.get('flatOrder'))
            flatOrder = string(char(data.get('flatOrder')));
        end
        shardsList(end+1).file = files(i).name; %#ok<AGROW>
        shardsList(end).num_samples = K;
        shardsList(end).start_idx   = total;
        total = total + K;
    end
    man.prefix=char(prefix);
    man.flatOrder=char(flatOrder);
    man.store_vec=store_vec;
    man.shards=shardsList;
    fid=fopen(manifestPath,'w'); fwrite(fid,jsonencode(man)); fclose(fid);
    fprintf("Built manifest with %d shard(s), total %d samples.\n", numel(files), total);
end
man = jsondecode(fileread(manifestPath));
flatOrder = string(man.flatOrder);
store_vec = logical(man.store_vec);
shards = arrayfun(@(x) fullfile(rootDir,x.file), man.shards, "UniformOutput",false);

%% ---------- Pass1: 计算 mean/std ----------
fprintf("Pass1: computing mean/std...\n");
muX=zeros(60,1); M2=zeros(60,1); count=0;
for s=1:numel(shards)
    data=py.numpy.load(shards{s}, pyargs("allow_pickle",true));
    if store_vec
        Xpy=data.get('X_vec60'); X=double(py.numpy.array(Xpy)); sh=cell(Xpy.shape);
        K=double(sh{2}); X=reshape(X,60,K);
    else
        XmPy=data.get('X_mats'); Xm=double(py.numpy.array(XmPy)); sh=cell(XmPy.shape);
        K=double(sh{4}); X=zeros(60,K);
        for k=1:K, X(:,k)=pack_5mats_to_vec60(Xm(:,:,:,k),flatOrder); end
    end
    for k=1:K
        count=count+1;
        delta=X(:,k)-muX;
        muX=muX+delta/count;
        M2=M2+delta.*(X(:,k)-muX);
    end
end
sigX=sqrt(max(M2/(count-1),1e-12));
fprintf("Mean/std computed from %d samples.\n", count);

%% ---------- 网络 ----------
layers=[
    featureInputLayer(60,"Normalization","none","Name","in")
    fullyConnectedLayer(256,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(256,"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(36,"Name","fc_out")
];
net=dlnetwork(layerGraph(layers));

%% ---------- 超参数 ----------
learnRate=1e-3; numEpochs=10; miniBatch=2048;
valRatio=0.1; % 每个 shard 中验证集比例
gradientDecayFactor=0.9; squaredGradientDecayFactor=0.999; eps=1e-8;
trailingAvg=[]; trailingAvgSq=[];
iter=0;

% 可选：早停（默认关闭）
useEarlyStop=true; patience=800; minDelta=0;  % 把 useEarlyStop=true 就开启
bestVal=inf; bestIter=0; bestEpoch=0; bestMarker=[];

%% ---------- 实时绘图 ----------
fig=figure('Name','Training Progress','NumberTitle','off');
ax=axes(fig); grid(ax,'on'); hold(ax,'on');
hTrain=animatedline('Color',[0.2 0.6 1],'LineWidth',1.5,'DisplayName','Train Loss');
hVal  =animatedline('Color',[1 0.3 0.3],'LineWidth',1.5,'DisplayName','Val Loss');
xlabel('Iteration'); ylabel('Loss'); title('Training & Validation Loss');
legend('show','Location','northeast');
drawnow;
logI=[]; logT=[]; logV=[];

%% ---------- 训练 ----------
sinceBest=0;
for e=1:numEpochs
    fprintf("Epoch %d/%d\n", e, numEpochs);
    ord=randperm(numel(shards));
    for si=1:numel(ord)
        data=py.numpy.load(shards{ord(si)}, pyargs("allow_pickle",true));
        if store_vec
            X=double(py.numpy.array(data.get('X_vec60')));
            Y=double(py.numpy.array(data.get('Y_vec36')));
            K=size(X,2);
        else
            XmPy=data.get('X_mats'); YmPy=data.get('Y_mats');
            Xm=double(py.numpy.array(XmPy)); Ym=double(py.numpy.array(YmPy));
            shX=cell(XmPy.shape); K=double(shX{4});
            X=zeros(60,K); Y=zeros(36,K);
            for k=1:K
                X(:,k)=pack_5mats_to_vec60(Xm(:,:,:,k),flatOrder);
                Y(:,k)=pack_3mats_to_vec36(Ym(:,:,:,k),flatOrder);
            end
        end
        X=(X-muX)./sigX;

        % 拆分 Train / Val
        idx=randperm(K);
        nVal=max(1, round(K*valRatio));
        valIdx=idx(1:nVal);
        trnIdx=idx(nVal+1:end);
        Xtr=X(:,trnIdx); Ytr=Y(:,trnIdx);
        Xv =X(:,valIdx); Yv =Y(:,valIdx);

        % mini-batch
        for t=1:miniBatch:size(Xtr,2)
            sel=t:min(t+miniBatch-1,size(Xtr,2));
            Xb=dlarray(single(Xtr(:,sel)),"CB");
            Yb=dlarray(single(Ytr(:,sel)),"CB");

            [grad,loss]=dlfeval(@modelGradients,net,Xb,Yb);
            [net,trailingAvg,trailingAvgSq]=adamupdate(net,grad,...
                trailingAvg,trailingAvgSq,iter+1,learnRate,...
                gradientDecayFactor,squaredGradientDecayFactor,eps);
            iter=iter+1;

            % 验证
            if mod(iter,50)==0 || t==1
                Yv_pred=forward(net,dlarray(single(Xv),"CB"));
                valLoss=mse(Yv_pred,dlarray(single(Yv),"CB"));
                trainLoss=double(gather(extractdata(loss)));
                valLoss=double(gather(extractdata(valLoss)));

                % 画图 & 记录
                addpoints(hTrain,iter,trainLoss);
                addpoints(hVal,iter,valLoss);
                drawnow limitrate
                logI(end+1)=iter; %#ok<AGROW>
                logT(end+1)=trainLoss; %#ok<AGROW>
                logV(end+1)=valLoss; %#ok<AGROW>
                fprintf("  iter=%d  train=%.6f  val=%.6f\n",iter,trainLoss,valLoss);

                % —— 更新最优点（标记 + checkpoint）——
                if valLoss < bestVal - minDelta
                    bestVal   = valLoss;
                    bestIter  = iter;
                    bestEpoch = e;
                    sinceBest = 0;

                    % 更新图上标记
                    if ~isempty(bestMarker) && isvalid(bestMarker)
                        delete(bestMarker);
                    end
                    bestMarker = plot(ax, bestIter, bestVal, 'o', ...
                        'MarkerSize',10,'LineWidth',1.5, ...
                        'MarkerEdgeColor',[0 0 0], 'MarkerFaceColor',[1 1 0], ...
                        'DisplayName','Best Val');

                    % 保存最佳 checkpoint
                    bestPath = fullfile(outDir, 'checkpoint_best.mat');
                    prep.muX = muX(:); prep.sigX = sigX(:); prep.flatOrder = char(flatOrder);
                    save(bestPath, 'net', 'prep', 'bestIter', 'bestVal', 'bestEpoch', '-v7.3');
                    fprintf("  ⭐ New best! val=%.6f @ iter=%d (epoch %d). Saved %s\n", bestVal, bestIter, bestEpoch, bestPath);
                else
                    sinceBest = sinceBest + 1;
                end

                % —— 可选早停 —— 
                if useEarlyStop && sinceBest >= patience
                    fprintf("Early stopping: no val improvement for %d checks.\n", patience);
                    break;
                end
            end
        end
        if useEarlyStop && sinceBest >= patience, break; end
    end
    if useEarlyStop && sinceBest >= patience, break; end
end

fprintf("Training done.\n");

% 保存曲线与日志
if isvalid(fig)
    saveas(fig, fullfile(outDir,'loss_curve.png'));
end
T=table(logI(:),logT(:),logV(:),'VariableNames',{'iteration','train_loss','val_loss'});
writetable(T, fullfile(outDir,'loss_log.csv'));

%% ---------- 导出最终模型 ----------
prep.muX=muX(:); prep.sigX=sigX(:); prep.flatOrder=char(flatOrder);
fid=fopen(fullfile(outDir,"deformer_mapper_prep.json"),'w'); fwrite(fid,jsonencode(prep)); fclose(fid);
save(fullfile(outDir,"deformer_mapper_dlnet.mat"),"net","-v7.3");

onnxPath=fullfile(outDir,"deformer_mapper.onnx");
try
    if exist('exportONNXNetwork','file')==2
        lgraph=layerGraph(layers); lgraph=assignLearnables(lgraph,net);
        exportONNXNetwork(lgraph,onnxPath,"OpsetVersion",17);
        fprintf("ONNX exported: %s\n",onnxPath);
    else
        fprintf("ONNX Converter 未安装，跳过导出。\n");
    end
catch ME
    warning("导出 ONNX 失败：%s",ME.message);
end
fprintf("✅ All done! Output: %s\n", outDir);
fprintf("   Best Val: %.6f at iter=%d (epoch %d)\n", bestVal, bestIter, bestEpoch);
end

%% ====== Helper Functions ======
function lgraph=assignLearnables(lgraph,dlnet)
params=dlnet.Learnables;
for i=1:size(params,1)
    layerName=params.Layer(i);
    paramName=params.Parameter(i);
    val=gather(extractdata(params.Value{i}));
    hit=strcmp({lgraph.Layers.Name},layerName);
    if any(hit)
        L=lgraph.Layers(hit);
        if isprop(L,paramName)
            L.(paramName)=val;
            lgraph=replaceLayer(lgraph,layerName,L);
        end
    end
end
end

function [gradients,loss]=modelGradients(net,Xb,Yb)
Yhat=forward(net,Xb);
loss=mse(Yhat,Yb);
gradients=dlgradient(loss,net.Learnables);
end

function vec60=pack_5mats_to_vec60(mats_4x4x5,flatOrder)
vec60=zeros(60,1); idx=1;
for k=1:5
    M=mats_4x4x5(:,:,k); M34=M(1:3,1:4);
    if flatOrder=="row", v=reshape(M34,12,1); else, v=reshape(M34.',12,1); end
    vec60(idx:idx+11)=v; idx=idx+12;
end
end

function vec36=pack_3mats_to_vec36(mats_4x4x3,flatOrder)
vec36=zeros(36,1); idx=1;
for k=1:3
    M=mats_4x4x3(:,:,k); M34=M(1:3,1:4);
    if flatOrder=="row", v=reshape(M34,12,1); else, v=reshape(M34.',12,1); end
    vec36(idx:idx+11)=v; idx=idx+12;
end
end
