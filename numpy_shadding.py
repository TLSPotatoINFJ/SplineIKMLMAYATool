# merge_npz_samples.py
import os, json, numpy as np

def merge_npz_samples(src_dir, dst_dir, prefix="merged", shard_size=10000):
    """
    合并单个 sample_*.npz 文件为大 npz shard：
    每个 shard 包含：
        X_vec60: (60, N)
        Y_vec36: (36, N)
        flatOrder: "row" (统一取第一个样本)
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npz")])
    if not files:
        print("❌ No npz found in", src_dir)
        return

    all_shards = []
    shard_idx = 0
    X_list, Y_list = [], []
    flat_order = "row"

    for i, fname in enumerate(files):
        fpath = os.path.join(src_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            if "flatOrder" in data:
                flat_order = str(data["flatOrder"])
            x = np.array(data["X_vec60"], dtype=np.float32).reshape(60, 1)
            y = np.array(data["Y_vec36"], dtype=np.float32).reshape(36, 1)
            X_list.append(x)
            Y_list.append(y)
        except Exception as e:
            print("⚠️ skip", fname, ":", e)
            continue

        # 存满一批就写出
        if len(X_list) >= shard_size or i == len(files) - 1:
            shard_idx += 1
            X_all = np.concatenate(X_list, axis=1)
            Y_all = np.concatenate(Y_list, axis=1)
            out_path = os.path.join(dst_dir, f"{prefix}_{shard_idx:06d}.npz")
            np.savez_compressed(out_path, X_vec60=X_all, Y_vec36=Y_all,
                                flatOrder=np.string_(flat_order))
            print(f"✅ Saved {out_path}  ({X_all.shape[1]} samples)")
            X_list, Y_list = [], []
            all_shards.append(out_path)

    # 生成 manifest
    manifest = {
        "prefix": prefix,
        "flatOrder": flat_order,
        "store_vec": True,
        "shards": [{"file": os.path.basename(p)} for p in all_shards]
    }
    with open(os.path.join(dst_dir, f"{prefix}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("📄 manifest saved.")

if __name__ == "__main__":
    merge_npz_samples(
        src_dir=r"D:\WKS\SamplesSingle",
        dst_dir=r"D:\WKS\SamplesMerged",
        prefix="shard",
        shard_size=10000
    )
