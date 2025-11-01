# deformer_mapper_ui.py
# Maya 2023+ | PySide2
from __future__ import annotations
import os, json, traceback, random, math, time
import numpy as np
from PySide2 import QtWidgets, QtCore, QtGui
import maya.cmds as cmds
from maya import mel  # ✅ 修复：使用 maya.mel.eval

# ====== onnxruntime（仅 Deformer Mapper 用；未装也可用 Random Motion） ======
try:
    import onnxruntime as ort
    _ORT_OK = True
except Exception as _e:
    _ORT_OK = False
    _ORT_ERR = _e

# ====== 公用 ======
def get_maya_main_window():
    import maya.OpenMayaUI as omui
    from shiboken2 import wrapInstance
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)

def _get_world_matrix(node: str) -> np.ndarray:
    m = cmds.xform(node, q=True, ws=True, m=True)
    return np.array(m, dtype=np.float64).reshape(4, 4)

def _pack_input_60(mats5, flat_order="row"):
    vecs = []
    for M in mats5:
        M34 = M[:3, :4]
        v = M34.reshape(-1) if flat_order == "row" else M34.T.reshape(-1)
        vecs.append(v.astype(np.float64))
    return np.concatenate(vecs, axis=0)

def _unpack_output_36_to_3mats(v36, flat_order="row"):
    assert v36.shape[-1] == 36
    mats = []
    for k in range(3):
        seg = v36[k*12:(k+1)*12]
        M34 = seg.reshape(3,4) if flat_order == "row" else seg.reshape(4,3).T
        mats.append(np.vstack([M34, np.array([0,0,0,1.0])]))
    return mats

def _vec60_to_5mats(vec60, flat_order="row"):
    assert vec60.shape[-1] == 60
    mats = []
    for k in range(5):
        seg = vec60[k*12:(k+1)*12]
        M34 = seg.reshape(3,4) if flat_order == "row" else seg.reshape(4,3).T
        mats.append(np.vstack([M34, np.array([0,0,0,1.0])]))
    return mats

# ====== 分片写入器（仅分片导出） ======
class ShardedNPZWriter:
    """
    把多条样本累积到内存，到达 shard_size 写出一个 .npz 分片：
      - store_vec=True: 存 X_vec60:(60,K), Y_vec36:(36,K)
      - store_vec=False: 存 X_mats:(4,4,5,K), Y_mats:(4,4,3,K)
    """
    def __init__(self, out_dir, prefix="shard", shard_size=10000,
                 flat_order="row", store_vec=True, compress=True):
        import numpy as _np
        self.np = _np
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = int(max(1, shard_size))
        self.flat_order = flat_order
        self.store_vec = store_vec
        self.compress = compress
        self.buf_X = []
        self.buf_Y = []
        self.sample_count = 0
        self.shard_index = 0
        os.makedirs(out_dir, exist_ok=True)
        self.manifest_path = os.path.join(out_dir, f"{prefix}_manifest.json")
        self._manifest = {
            "prefix": prefix,
            "shard_size": self.shard_size,
            "flatOrder": flat_order,
            "store_vec": store_vec,
            "compress": compress,
            "shards": []  # [{file,num_samples,start_idx}]
        }

    def append_vec(self, X_vec60, Y_vec36):
        self.buf_X.append(self.np.asarray(X_vec60, dtype=self.np.float32).reshape(60))
        self.buf_Y.append(self.np.asarray(Y_vec36, dtype=self.np.float32).reshape(36))
        self.sample_count += 1
        if len(self.buf_X) >= self.shard_size:
            self._flush()

    def append_mats(self, X_mats, Y_mats):
        self.buf_X.append(self.np.asarray(X_mats, dtype=self.np.float32).reshape(4,4,5))
        self.buf_Y.append(self.np.asarray(Y_mats, dtype=self.np.float32).reshape(4,4,3))
        self.sample_count += 1
        if len(self.buf_X) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buf_X:
            return
        K = len(self.buf_X)
        fname = f"{self.prefix}_{self.shard_index:06d}.npz"
        fpath = os.path.join(self.out_dir, fname)
        save = self.np.savez_compressed if self.compress else self.np.savez
        if self.store_vec:
            X_all = self.np.stack(self.buf_X, axis=1)  # (60,K)
            Y_all = self.np.stack(self.buf_Y, axis=1)  # (36,K)
            save(fpath, X_vec60=X_all, Y_vec36=Y_all, flatOrder=self.flat_order)
        else:
            X_all = self.np.stack(self.buf_X, axis=2)  # (4,4,5,K)
            Y_all = self.np.stack(self.buf_Y, axis=2)  # (4,4,3,K)
            save(fpath, X_mats=X_all, Y_mats=Y_all, flatOrder=self.flat_order)
        self._manifest["shards"].append({
            "file": fname,
            "num_samples": K,
            "start_idx": self.sample_count - K
        })
        self.buf_X.clear(); self.buf_Y.clear()
        self.shard_index += 1
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2)

    def close(self):
        self._flush()

# ====== Deformer Mapper（ONNX 推理） ======
class DeformerMapperSession:
    def __init__(self, onnx_path: str, prep_json: str, use_gpu=True):
        if not _ORT_OK:
            raise RuntimeError(
                "未找到 onnxruntime，请安装：\n"
                '"%MAYA_LOCATION%\\bin\\mayapy.exe" -m pip install onnxruntime'
            ) from _ORT_ERR
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在：{onnx_path}")
        if not os.path.isfile(prep_json):
            raise FileNotFoundError(f"预处理 JSON 不存在：{prep_json}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

        with open(prep_json, "r", encoding="utf-8") as f:
            prep = json.load(f)
        self.muX  = np.array(prep["muX"], dtype=np.float64).reshape(-1)
        self.sigX = np.array(prep["sigX"], dtype=np.float64).reshape(-1)
        self.flat_order = prep.get("flatOrder", "row")
        if self.muX.shape[0] != 60 or self.sigX.shape[0] != 60:
            raise ValueError("prep.json 中 muX/sigX 维度必须为 60。")

        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def predict_from_nodes(self, input_nodes):
        if len(input_nodes) != 5:
            raise ValueError("需要 5 个输入节点。")
        mats5 = [_get_world_matrix(n) for n in input_nodes]
        x60 = _pack_input_60(mats5, self.flat_order)
        xn = (x60 - self.muX) / self.sigX
        y = self.sess.run([self.out_name], {self.in_name: xn.reshape(1,-1).astype(np.float32)})[0]
        return _unpack_output_36_to_3mats(y.reshape(-1).astype(np.float64), self.flat_order)

    def apply_to_nodes(self, input_nodes, output_nodes):
        if len(output_nodes) != 3:
            raise ValueError("需要 3 个输出节点。")
        mats3 = self.predict_from_nodes(input_nodes)
        for node, M in zip(output_nodes, mats3):
            cmds.xform(node, ws=True, m=M.reshape(-1).tolist())
        return mats3

# ====== Random Motion（严格校验属性随机器 + 仅分片导出 + Stop） ======
class RandomMotionPanel(QtWidgets.QWidget):
    """
    - 随机位姿
    - 属性随机器（严格校验）
    - 仅分片导出（Sharded NPZ）
    - Stop 按钮：可中断循环，安全关闭 writer
    """
    def __init__(self, parent=None, log_cb=None,
                 get_nodes_cb=None, get_sample_cb=None):
        super().__init__(parent)
        self.target_object = None
        self.radius = 5.0
        self.center_pos = [0, 0, 0]
        self.iterations = 1
        self._log = log_cb or (lambda msg, level="info": None)
        self._get_nodes_cb = get_nodes_cb
        self._get_sample_cb = get_sample_cb
        self._writer = None
        self._stop_flag = False
        self._build_ui()

    # ---------- plug 解析 ----------
    def _resolve_plug(self, node: str, attr: str) -> str | None:
        attr = attr.strip()
        if not node and "." in attr and cmds.objExists(attr):
            return attr  # 已是完整 plug
        if node:
            plug = f"{node}.{attr}"
            if cmds.objExists(plug):
                return plug
            if cmds.objExists(node):
                shapes = cmds.listRelatives(node, s=True, ni=True, f=False) or []
                for shp in shapes:
                    shp_plug = f"{shp}.{attr}"
                    if cmds.objExists(shp_plug):
                        return shp_plug
        if "." in attr and cmds.objExists(attr):
            return attr
        return None

    # ---------- UI ----------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 目标物体
        h = QtWidgets.QHBoxLayout()
        self.object_line = QtWidgets.QLineEdit(); self.object_line.setPlaceholderText("Enter object name or select it")
        self.object_line.returnPressed.connect(self.load_object_by_name)
        btn_sel = QtWidgets.QPushButton("Select from Scene"); btn_sel.clicked.connect(self.select_object)
        btn_ref = QtWidgets.QPushButton("Refresh Selection"); btn_ref.clicked.connect(self.select_object)
        h.addWidget(QtWidgets.QLabel("Target Object:"))
        h.addWidget(self.object_line); h.addWidget(btn_sel); h.addWidget(btn_ref)
        layout.addLayout(h)

        # 半径 / 迭代
        h2 = QtWidgets.QHBoxLayout()
        self.radius_spin = QtWidgets.QDoubleSpinBox(); self.radius_spin.setRange(0.0, 1e9); self.radius_spin.setValue(self.radius); self.radius_spin.valueChanged.connect(self.update_radius)
        self.iterations_spin = QtWidgets.QSpinBox(); self.iterations_spin.setRange(1, 1000000); self.iterations_spin.setValue(self.iterations); self.iterations_spin.valueChanged.connect(self.update_iterations)
        h2.addWidget(QtWidgets.QLabel("Radius:")); h2.addWidget(self.radius_spin)
        h2.addWidget(QtWidgets.QLabel("Iterations:")); h2.addWidget(self.iterations_spin)
        layout.addLayout(h2)

        # 中心位置
        h3 = QtWidgets.QHBoxLayout()
        self.center_x = QtWidgets.QDoubleSpinBox(); self.center_x.setRange(-1e9, 1e9)
        self.center_y = QtWidgets.QDoubleSpinBox(); self.center_y.setRange(-1e9, 1e9)
        self.center_z = QtWidgets.QDoubleSpinBox(); self.center_z.setRange(-1e9, 1e9)
        h3.addWidget(QtWidgets.QLabel("Center (auto-set):"))
        h3.addWidget(QtWidgets.QLabel("X:")); h3.addWidget(self.center_x)
        h3.addWidget(QtWidgets.QLabel("Y:")); h3.addWidget(self.center_y)
        h3.addWidget(QtWidgets.QLabel("Z:")); h3.addWidget(self.center_z)
        layout.addLayout(h3)
        self.update_center_btn = QtWidgets.QPushButton("Update Center to Pivot")
        self.update_center_btn.setEnabled(False)
        self.update_center_btn.clicked.connect(self.update_center)
        layout.addWidget(self.update_center_btn)

        # 分片导出设置（仅分片，无单文件）
        exp_box = QtWidgets.QGroupBox("Sharded Export (.npz)")
        gl = QtWidgets.QGridLayout(exp_box)
        self.export_chk = QtWidgets.QCheckBox("Enable"); self.export_chk.setChecked(False)

        self.dir_edit = QtWidgets.QLineEdit()
        btn_dir = QtWidgets.QPushButton("Browse…"); btn_dir.clicked.connect(self._browse_dir)
        self.prefix_edit = QtWidgets.QLineEdit("shard")

        self.shard_size_spin = QtWidgets.QSpinBox(); self.shard_size_spin.setRange(100, 10000000); self.shard_size_spin.setValue(10000)
        self.compress_chk = QtWidgets.QCheckBox("Compress NPZ"); self.compress_chk.setChecked(True)
        self.store_vec_chk = QtWidgets.QCheckBox("Store as vectors (X_vec/Y_vec)"); self.store_vec_chk.setChecked(True)

        gl.addWidget(self.export_chk, 0, 0)
        gl.addWidget(QtWidgets.QLabel("Export dir:"), 1, 0); gl.addWidget(self.dir_edit, 1, 1); gl.addWidget(btn_dir, 1, 2)
        gl.addWidget(QtWidgets.QLabel("Prefix:"), 2, 0); gl.addWidget(self.prefix_edit, 2, 1, 1, 2)
        gl.addWidget(QtWidgets.QLabel("Shard Size:"), 3, 0); gl.addWidget(self.shard_size_spin, 3, 1)
        gl.addWidget(self.compress_chk, 3, 2)
        gl.addWidget(self.store_vec_chk, 4, 1, 1, 2)
        layout.addWidget(exp_box)

        # Attribute Randomizer
        attr_box = QtWidgets.QGroupBox("Attribute Randomizer (strict validated)")
        v = QtWidgets.QVBoxLayout(attr_box)
        th = QtWidgets.QHBoxLayout()
        self.attr_name_edit = QtWidgets.QLineEdit(); self.attr_name_edit.setPlaceholderText("Attr or full plug (e.g., endAngle / polyPipeShape1.endAngle)")
        self.attr_min_spin = QtWidgets.QDoubleSpinBox(); self.attr_min_spin.setRange(-1e12, 1e12); self.attr_min_spin.setValue(0.0)
        self.attr_max_spin = QtWidgets.QDoubleSpinBox(); self.attr_max_spin.setRange(-1e12, 1e12); self.attr_max_spin.setValue(1.0)
        btn_add_sel   = QtWidgets.QPushButton("Add Rows from Selection"); btn_add_sel.clicked.connect(self._add_rows_from_selection)
        btn_add_chbox = QtWidgets.QPushButton("Add From Channel Box"); btn_add_chbox.clicked.connect(self._add_from_channel_box)
        btn_remove    = QtWidgets.QPushButton("Remove Selected"); btn_remove.clicked.connect(self._remove_selected_rows)
        btn_probe     = QtWidgets.QPushButton("Probe Rules"); btn_probe.clicked.connect(self._probe_rules)
        th.addWidget(QtWidgets.QLabel("Attr/Plug:")); th.addWidget(self.attr_name_edit, 2)
        th.addWidget(QtWidgets.QLabel("Min:")); th.addWidget(self.attr_min_spin)
        th.addWidget(QtWidgets.QLabel("Max:")); th.addWidget(self.attr_max_spin)
        th.addWidget(btn_add_sel); th.addWidget(btn_add_chbox); th.addWidget(btn_remove); th.addWidget(btn_probe)
        v.addLayout(th)

        self.attr_table = QtWidgets.QTableWidget(0, 5)
        self.attr_table.setHorizontalHeaderLabels(["Node", "Attr/Plug", "Min", "Max", "Resolved Plug"])
        self.attr_table.horizontalHeader().setStretchLastSection(True)
        self.attr_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.attr_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.attr_table)
        layout.addWidget(attr_box)

        # 执行按钮与状态（含 Stop）
        btns = QtWidgets.QHBoxLayout()
        self.random_btn = QtWidgets.QPushButton("Start Random Move")
        self.random_btn.setEnabled(False)
        self.random_btn.clicked.connect(self.random_move)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btns.addWidget(self.random_btn); btns.addWidget(self.stop_btn)
        layout.addLayout(btns)

        self.status_label = QtWidgets.QLabel("No object selected. Select and click 'Select from Scene'.")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    # ---------- 基础逻辑 ----------
    def _browse_dir(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Export Directory", "")
        if p: self.dir_edit.setText(p)

    def load_object_by_name(self):
        obj_name = self.object_line.text().strip()
        if cmds.objExists(obj_name):
            self.target_object = obj_name
            self.random_btn.setEnabled(True); self.update_center_btn.setEnabled(True)
            self.update_center()
            self.status_label.setText(f"Loaded: {self.target_object}")
            self._log(f"RandomMotion: Loaded {self.target_object}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Object '{obj_name}' not found.")

    def select_object(self):
        sel = cmds.ls(sl=True, transforms=True)
        if sel:
            self.target_object = sel[0]
            self.object_line.setText(self.target_object)
            self.random_btn.setEnabled(True); self.update_center_btn.setEnabled(True)
            self.update_center()
            self.status_label.setText(f"Selected: {self.target_object}")
            self._log(f"RandomMotion: Selected {self.target_object}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an object in the scene first.")

    def update_radius(self, value): self.radius = value
    def update_iterations(self, value): self.iterations = value

    def update_center(self):
        if self.target_object:
            try:
                self.center_pos = cmds.xform(self.target_object, q=True, ws=True, t=True)
                self.center_x.setValue(self.center_pos[0])
                self.center_y.setValue(self.center_pos[1])
                self.center_z.setValue(self.center_pos[2])
                self.status_label.setText(f"Center updated to: {self.center_pos}")
            except Exception as e:
                cmds.warning(f"Error updating center: {e}")
                self.status_label.setText(f"Error: {e}")

    # ---------- 规则增删查 ----------
    def _insert_rule_row(self, node, shown_attr, mn, mx, resolved_plug):
        r = self.attr_table.rowCount()
        self.attr_table.insertRow(r)
        for c, text in enumerate([node, shown_attr, str(mn), str(mx), resolved_plug]):
            item = QtWidgets.QTableWidgetItem(text)
            self.attr_table.setItem(r, c, item)

    def _is_supported_attr(self, plug: str) -> bool:
        try:
            atype = cmds.getAttr(plug, type=True)
            return atype in ("double","float","long","short","byte","bool","enum","doubleAngle","doubleLinear","double3")
        except Exception:
            return False

    def _add_rows_from_selection(self):
        raw = self.attr_name_edit.text().strip()
        if not raw:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter attribute name or full plug.")
            return
        sel = cmds.ls(sl=True) or []
        if not sel:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select objects in scene.")
            return
        mn, mx = self.attr_min_spin.value(), self.attr_max_spin.value()
        if mn > mx: mn, mx = mx, mn

        skipped, added = [], 0
        for node in sel:
            plug = self._resolve_plug(node, raw)
            if plug is None or not self._is_supported_attr(plug):
                skipped.append(f"{node}.{raw}")
                continue
            self._insert_rule_row(node, raw, mn, mx, plug); added += 1

        if added == 0:
            QtWidgets.QMessageBox.critical(self, "No Valid Attribute",
                "No valid attributes were found on selection.\n\n" + "\n".join(skipped[:20]))
        else:
            msg = f"Added {added} row(s)."
            if skipped: msg += "\nSkipped:\n  - " + "\n  - ".join(skipped[:20])
            QtWidgets.QMessageBox.information(self, "Add Results", msg)
            self._log("[AddRows] " + msg)

    def _add_from_channel_box(self):
        try:
            ch_win = mel.eval('$tmp=$gChannelBoxName')  # ✅ 修复：使用 maya.mel
            sels = cmds.channelBox(ch_win, q=True, sma=True) or []
            obj  = (cmds.channelBox(ch_win, q=True, mol=True) or [None])[0]
        except Exception:
            sels, obj = [], None
        if not sels or not obj:
            QtWidgets.QMessageBox.warning(self, "Warning", "No attribute is selected in Channel Box."); return

        mn, mx = self.attr_min_spin.value(), self.attr_max_spin.value()
        if mn > mx: mn, mx = mx, mn
        added, skipped = 0, []
        for attr in sels:
            plug = self._resolve_plug(obj, attr)
            if plug is None or not self._is_supported_attr(plug):
                skipped.append(f"{obj}.{attr}"); continue
            self._insert_rule_row(obj, attr, mn, mx, plug); added += 1

        if added == 0:
            QtWidgets.QMessageBox.critical(self, "No Valid Attribute",
                "No valid channel-box attributes were added.\n\n" + "\n".join(skipped[:20]))
        else:
            msg = f"Added {added} row(s) from Channel Box."
            if skipped: msg += "\nSkipped:\n  - " + "\n".join(skipped[:20])
            QtWidgets.QMessageBox.information(self, "Add Results", msg)
            self._log("[AddFromChannelBox] " + msg)

    def _remove_selected_rows(self):
        rows = sorted({i.row() for i in self.attr_table.selectedIndexes()}, reverse=True)
        for r in rows: self.attr_table.removeRow(r)

    def _gather_attr_rules(self):
        rules = []
        for r in range(self.attr_table.rowCount()):
            mn = float(self.attr_table.item(r, 2).text())
            mx = float(self.attr_table.item(r, 3).text())
            if mn > mx: mn, mx = mx, mn
            plug = self.attr_table.item(r, 4).text().strip()
            if cmds.objExists(plug):
                rules.append({"plug": plug, "min": mn, "max": mx})
        return rules

    def _probe_rules(self):
        rules = self._gather_attr_rules()
        if not rules:
            self._log("[Probe] No rules.", "warn"); return
        for rule in rules:
            plug = rule["plug"]
            try:
                con = cmds.listConnections(plug, s=True, d=False, p=True) or []
                locked = cmds.getAttr(plug, lock=True)
                atype = cmds.getAttr(plug, type=True)
                try: val = cmds.getAttr(plug)
                except: val = "<unreadable>"
                self._log(f"[Probe] {plug} | type={atype} | locked={locked} | connected={'Yes' if con else 'No'} | value={val}")
            except Exception as e:
                self._log(f"[Probe] {plug} error: {e}", "error")

    # ---------- Stop ----------
    def _on_stop_clicked(self):
        self._stop_flag = True
        self.stop_btn.setEnabled(False)
        self.status_label.setText("⛔ Stop requested... finishing current iteration.")
        self._log("用户请求中断 Random Move。", "warn")

    def _set_running_ui(self, running: bool):
        # 运行时禁用可编辑控件，防误操作
        for w in [self.object_line, self.radius_spin, self.iterations_spin,
                  self.update_center_btn, self.export_chk, self.dir_edit,
                  self.prefix_edit, self.shard_size_spin, self.compress_chk, self.store_vec_chk,
                  self.attr_name_edit, self.attr_min_spin, self.attr_max_spin, self.attr_table]:
            w.setEnabled(not running)
        self.random_btn.setEnabled(not running)  # ✅ 修复：去掉 !running
        self.stop_btn.setEnabled(running)
        QtWidgets.QApplication.processEvents()

    # ---------- 随机移动 & 仅分片导出（支持中断） ----------
    def random_move(self):
        if not self.target_object:
            QtWidgets.QMessageBox.warning(self, "Warning", "No target object selected."); return

        self._stop_flag = False
        self._set_running_ui(True)
        self.status_label.setText("Running... (Sharded export). Click Stop to interrupt.")

        try:
            self.center_pos = [self.center_x.value(), self.center_y.value(), self.center_z.value()]
            do_export = self.export_chk.isChecked()
            export_dir = self.dir_edit.text().strip()
            prefix = self.prefix_edit.text().strip() or "shard"
            shard_size = self.shard_size_spin.value()
            compress = self.compress_chk.isChecked()
            store_vec = self.store_vec_chk.isChecked()

            if do_export:
                if not export_dir:
                    QtWidgets.QMessageBox.warning(self, "Warning", "Please choose an export directory."); 
                    self._set_running_ui(False); return
                os.makedirs(export_dir, exist_ok=True)
                self._writer = None  # 延迟到首次 sample 才创建，拿 flatOrder

            rules = self._gather_attr_rules()
            last_pos = None

            for it in range(1, self.iterations + 1):
                if self._stop_flag:
                    self._log("🟠 中断信号收到，停止循环。", "warn")
                    break

                # 1) 随机位姿
                r = self.radius * random.random()
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                x_off = r * math.sin(phi) * math.cos(theta)
                y_off = r * math.sin(phi) * math.sin(theta)
                z_off = r * math.cos(phi)
                new_pos = [self.center_pos[0]+x_off, self.center_pos[1]+y_off, self.center_pos[2]+z_off]
                new_rot = [random.uniform(-180,180), random.uniform(-180,180), random.uniform(-180,180)]
                cmds.xform(self.target_object, ws=True, t=new_pos)
                cmds.xform(self.target_object, ws=True, ro=new_rot)
                last_pos = new_pos

                # 2) 属性随机
                if rules:
                    self._apply_random_attrs(rules)

                cmds.refresh()

                # 3) 导出样本（仅分片）
                if do_export and self._get_sample_cb:
                    sample = self._get_sample_cb()
                    if not sample:
                        self._log("Export skipped: sample callback returned None.", "warn")
                    else:
                        flatOrder = sample.get("flatOrder","row")
                        if self._writer is None:
                            self._writer = ShardedNPZWriter(
                                out_dir=export_dir,
                                prefix=prefix,
                                shard_size=shard_size,
                                flat_order=flatOrder,
                                store_vec=store_vec,
                                compress=compress
                            )
                            self._log(f"[Shard] Writer created: size={shard_size}, compress={compress}, store_vec={store_vec}, flatOrder={flatOrder}")
                        if store_vec:
                            self._writer.append_vec(sample["X_vec60"], sample["Y_vec36"])
                        else:
                            self._writer.append_mats(sample["X_mats"], sample["Y_mats"])

                # 每 10 次刷新一次 UI
                if it % 10 == 0:
                    self.status_label.setText(f"Iteration {it}/{self.iterations}")
                    QtWidgets.QApplication.processEvents()

            # 结束：关闭 writer
            if self._writer is not None:
                try:
                    self._writer.close()
                    self._log("[Shard] Writer closed.")
                except Exception as e:
                    self._log(f"[Shard] Close error: {e}", "error")
                self._writer = None

            self.status_label.setText(f"Completed or Stopped at {it}/{self.iterations}. Final pos: {last_pos}")
            self._log(f"RandomMotion: Done/Stopped at {it}/{self.iterations}. Final pos: {last_pos}")

        except Exception:
            self._log("RandomMove failed:\n" + traceback.format_exc(), "error")

        finally:
            self._set_running_ui(False)
            self._stop_flag = False
            QtWidgets.QApplication.processEvents()

    def _apply_random_attrs(self, rules):
        for rule in rules:
            plug, vmin, vmax = rule["plug"], rule["min"], rule["max"]
            try:
                if cmds.getAttr(plug, lock=True):
                    self._log(f"[Attr] Locked: {plug}", "warn"); continue
                con = cmds.listConnections(plug, s=True, d=False, p=True) or []
                if con:
                    self._log(f"[Attr] Skipped connected plug: {plug} <- {con[0]}", "warn"); continue
                atype = cmds.getAttr(plug, type=True)
                val = random.uniform(vmin, vmax)

                if atype in ("long","short","byte","bool","enum"):
                    val = int(round(val)); cmds.setAttr(plug, val)
                elif atype in ("double","float","doubleAngle","doubleLinear"):
                    cmds.setAttr(plug, float(val))
                elif atype == "double3":
                    v3 = [random.uniform(vmin, vmax) for _ in range(3)]
                    try:
                        cmds.setAttr(plug, *v3, type="double3")
                    except Exception:
                        for axis, vv in zip(("X","Y","Z"), v3):
                            sub = plug + axis if not plug.endswith(("X","Y","Z")) else plug
                            try: cmds.setAttr(sub, vv)
                            except Exception as e2: self._log(f"[Attr] Set failed {sub}: {e2}", "warn")
                        continue
                else:
                    self._log(f"[Attr] Unsupported type '{atype}' on {plug}", "warn"); continue
                self._log(f"[Attr] {plug} set to {val if atype!='double3' else '[...]'}")
            except Exception as e:
                self._log(f"[Attr] Set failed {plug}: {e}", "error")

    def showEvent(self, event):
        try:
            sel = cmds.ls(sl=True, transforms=True) or []
            if sel:
                self.target_object = sel[0]
                self.object_line.setText(self.target_object)
                self.random_btn.setEnabled(True)
                self.update_center_btn.setEnabled(True)
                self.update_center()
        except Exception:
            pass
        super().showEvent(event)

# ====== 主窗口 ======
class DeformerMapperUI(QtWidgets.QDialog):
    WINDOW_TITLE = "Deformer Tools (Mapper + Random Motion + Sharded Export + Stop)"

    def __init__(self, parent=None):
        super().__init__(parent or get_maya_main_window())
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumWidth(960)
        self.session: DeformerMapperSession | None = None
        self._build_ui()

    # 日志
    def _log(self, msg: str, level: str = "info"):
        prefix = {"info":"[INFO] ", "warn":"[WARN] ", "error":"[ERROR] "}.get(level, "[INFO] ")
        self.log_edit.appendPlainText(prefix + msg)
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_edit.setTextCursor(cursor)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        # ---- Tab1: Deformer Mapper ----
        deformer_tab = QtWidgets.QWidget(); dlay = QtWidgets.QVBoxLayout(deformer_tab)
        path_box = QtWidgets.QGroupBox("模型与预处理"); gl = QtWidgets.QGridLayout(path_box)
        self.model_edit = QtWidgets.QLineEdit()
        self.prep_edit  = QtWidgets.QLineEdit()
        b1 = QtWidgets.QPushButton("选择 ONNX…"); b1.clicked.connect(self._browse_onnx)
        b2 = QtWidgets.QPushButton("选择 JSON…"); b2.clicked.connect(self._browse_json)
        gl.addWidget(QtWidgets.QLabel("ONNX 模型"), 0,0); gl.addWidget(self.model_edit, 0,1); gl.addWidget(b1, 0,2)
        gl.addWidget(QtWidgets.QLabel("预处理 JSON"), 1,0); gl.addWidget(self.prep_edit, 1,1); gl.addWidget(b2, 1,2)
        dlay.addWidget(path_box)

        node_box = QtWidgets.QGroupBox("节点"); nl = QtWidgets.QGridLayout(node_box)
        self.inputs_edits  = [QtWidgets.QLineEdit() for _ in range(5)]
        self.outputs_edits = [QtWidgets.QLineEdit() for _ in range(3)]
        fill_in  = QtWidgets.QPushButton("用当前选择填充 5 个输入");  fill_in.clicked.connect(self._fill_inputs_from_selection)
        fill_out = QtWidgets.QPushButton("用当前选择填充 3 个输出"); fill_out.clicked.connect(self._fill_outputs_from_selection)
        nl.addWidget(QtWidgets.QLabel("输入节点 (5)"), 0,0,1,3)
        for i,e in enumerate(self.inputs_edits):
            nl.addWidget(QtWidgets.QLabel(f"In{i+1}"), i+1,0); nl.addWidget(e, i+1,1,1,2)
        row0 = len(self.inputs_edits)+1; nl.addWidget(fill_in, row0,1,1,2)
        row1 = row0+1; nl.addWidget(QtWidgets.QLabel("输出节点 (3)"), row1,0,1,3)
        for j,e in enumerate(self.outputs_edits):
            nl.addWidget(QtWidgets.QLabel(f"Out{j+1}"), row1+j+1,0); nl.addWidget(e, row1+j+1,1,1,2)
        row2 = row1+len(self.outputs_edits)+1; nl.addWidget(fill_out, row2,1,1,2)
        dlay.addWidget(node_box)

        opt_box = QtWidgets.QGroupBox("选项"); oh = QtWidgets.QHBoxLayout(opt_box)
        self.use_gpu_chk   = QtWidgets.QCheckBox("使用 GPU (若可用)"); self.use_gpu_chk.setChecked(True)
        self.keep_open_chk = QtWidgets.QCheckBox("推理后保持窗口");     self.keep_open_chk.setChecked(True)
        oh.addWidget(self.use_gpu_chk); oh.addWidget(self.keep_open_chk); oh.addStretch(1)
        dlay.addWidget(opt_box)

        btn_h = QtWidgets.QHBoxLayout()
        self.run_btn    = QtWidgets.QPushButton("推理并应用到输出"); self.run_btn.clicked.connect(self._on_run)
        self.reload_btn = QtWidgets.QPushButton("仅加载/重载模型");   self.reload_btn.clicked.connect(self._on_reload)
        self.export_btn = QtWidgets.QPushButton("导出当前 X/Y 样本 (.npz)"); self.export_btn.clicked.connect(self._on_export_sample_dialog)
        self.load_btn   = QtWidgets.QPushButton("载入样本并写回场景 (.npz)"); self.load_btn.clicked.connect(self._on_load_sample_dialog)
        btn_h.addWidget(self.run_btn); btn_h.addWidget(self.reload_btn); btn_h.addWidget(self.export_btn); btn_h.addWidget(self.load_btn)
        dlay.addLayout(btn_h)

        self.tabs.addTab(deformer_tab, "Deformer Mapper")

        # ---- Tab2: Random Motion （仅分片导出 + Stop）----
        self.rand_tab = RandomMotionPanel(
            log_cb=self._log,
            get_nodes_cb=self._get_nodes_for_export,
            get_sample_cb=self._get_current_sample
        )
        self.tabs.addTab(self.rand_tab, "Random Motion")

        # 底部日志
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(1500)
        root.addWidget(self.log_edit, 1)

        if not _ORT_OK:
            self._log("未检测到 onnxruntime：如需推理请安装 -> mayapy -m pip install onnxruntime", "warn")

    # ---- Deformer 逻辑 ----
    def _browse_onnx(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 ONNX 模型", "", "ONNX (*.onnx)")
        if p: self.model_edit.setText(p)

    def _browse_json(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 预处理 JSON", "", "JSON (*.json)")
        if p: self.prep_edit.setText(p)

    def _fill_inputs_from_selection(self):
        sel = cmds.ls(sl=True, long=False) or []
        if len(sel) < 5:
            self._log(f"请选择至少 5 个节点（当前 {len(sel)} 个）。", "warn"); return
        for i in range(5): self.inputs_edits[i].setText(sel[i])
        self._log(f"已用选择填充输入：{sel[:5]}")

    def _fill_outputs_from_selection(self):
        sel = cmds.ls(sl=True, long=False) or []
        if len(sel) < 3:
            self._log(f"请选择至少 3 个节点（当前 {len(sel)} 个）。", "warn"); return
        for i in range(3): self.outputs_edits[i].setText(sel[i])
        self._log(f"已用选择填充输出：{sel[:3]}")

    def _on_reload(self):
        try:
            self.session = self._build_session()
            self._log("模型加载成功 ✅")
        except Exception as e:
            self._log("模型加载失败：\n" + "".join(traceback.format_exception_only(type(e), e)), "error")

    def _on_run(self):
        try:
            if self.session is None:
                self.session = self._build_session()
            inputs, outputs = self._get_nodes_for_export()
            if not all(inputs) or not all(outputs):
                self._log("输入/输出节点未填满。", "warn"); return
            mats = self.session.apply_to_nodes(inputs, outputs)
            self._log("推理完成并已写回矩阵：")
            for i, M in enumerate(mats, 1):
                self._log(f"Out{i}:\n{np.array2string(M, formatter={'float_kind':lambda x: f'{x: .6f}'})}")
            if not self.keep_open_chk.isChecked():
                self.close()
        except Exception:
            self._log("推理失败：\n" + traceback.format_exc(), "error")

    def _build_session(self):
        onnx_path = self.model_edit.text().strip()
        prep_path = self.prep_edit.text().strip()
        if not onnx_path: raise RuntimeError("请先选择 ONNX 模型路径。")
        if not prep_path: raise RuntimeError("请先选择 预处理 JSON 路径。")
        use_gpu = self.use_gpu_chk.isChecked()
        return DeformerMapperSession(onnx_path, prep_path, use_gpu)

    # ---- 导出/载入（调试用的单样本） ----
    def _on_export_sample_dialog(self):
        try:
            sample = self._get_current_sample()
            if not sample:
                self._log("无法获取当前样本（请确认 5个输入/3个输出已填写）。", "warn"); return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存 X/Y 样本为 .npz（调试用）", "sample_xy.npz", "NumPy Zip (*.npz)")
            if not path: self._log("已取消导出。"); return
            if not path.lower().endswith(".npz"): path += ".npz"
            self._export_sample_to_path_dict(sample, path)
            self._log(f"✅ 已导出样本：{path}")
        except Exception:
            self._log("导出失败：\n" + traceback.format_exc(), "error")

    def _on_load_sample_dialog(self):
        try:
            inputs, outputs = self._get_nodes_for_export()
            if not all(inputs) or not all(outputs):
                self._log("输入/输出节点未填满，无法载入样本。", "warn"); return
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择样本 .npz（调试用）", "", "NumPy Zip (*.npz)")
            if not path: self._log("已取消载入。"); return

            data = np.load(path, allow_pickle=True)
            flat_order = str(data["flatOrder"]) if "flatOrder" in data else "row"

            # X
            if "X_mats" in data:
                X_mats = data["X_mats"]; mats5 = [X_mats[:,:,i] for i in range(5)]
            elif "X_vec60" in data:
                mats5 = _vec60_to_5mats(np.array(data["X_vec60"]).reshape(-1), flat_order)
            else:
                raise ValueError("样本中没有 X。")

            # Y
            if "Y_mats" in data:
                Y_mats = data["Y_mats"]; mats3 = [Y_mats[:,:,i] for i in range(3)]
            elif "Y_vec36" in data:
                mats3 = _unpack_output_36_to_3mats(np.array(data["Y_vec36"]).reshape(-1), flat_order)
            else:
                raise ValueError("样本中没有 Y。")

            for node, M in zip(inputs, mats5):
                cmds.xform(node, ws=True, m=M.reshape(-1).tolist())
            for node, M in zip(outputs, mats3):
                cmds.xform(node, ws=True, m=M.reshape(-1).tolist())
            self._log(f"✅ 已载入样本并写回：{path}")
        except Exception:
            self._log("载入失败：\n" + traceback.format_exc(), "error")

    # ---- Random Motion 回调：获取节点 / 当前样本 ----
    def _get_nodes_for_export(self):
        inputs  = [e.text().strip() for e in self.inputs_edits]
        outputs = [e.text().strip() for e in self.outputs_edits]
        return inputs, outputs

    def _get_current_sample(self):
        """返回 dict：{X_mats(4,4,5), Y_mats(4,4,3), X_vec60(60,), Y_vec36(36,), flatOrder}"""
        inputs, outputs = self._get_nodes_for_export()
        if not (len(inputs)==5 and len(outputs)==3 and all(inputs) and all(outputs)):
            return None
        mats5 = [_get_world_matrix(n) for n in inputs]
        mats3 = [_get_world_matrix(n) for n in outputs]
        X_mats = np.stack(mats5, axis=-1)   # (4,4,5)
        Y_mats = np.stack(mats3, axis=-1)   # (4,4,3)
        flatOrder = "row"
        X_vec60 = _pack_input_60(mats5, flatOrder)
        Y_vec36 = []
        for M in mats3:
            M34 = M[:3, :4]
            Y_vec36.extend(M34.reshape(-1) if flatOrder == "row" else M34.T.reshape(-1))
        Y_vec36 = np.asarray(Y_vec36, dtype=np.float64)
        return {
            "X_mats": X_mats, "Y_mats": Y_mats,
            "X_vec60": X_vec60, "Y_vec36": Y_vec36,
            "flatOrder": flatOrder
        }

    def _export_sample_to_path_dict(self, sample_dict, path):
        flatOrder = sample_dict.get("flatOrder", "row")
        X_mats = sample_dict["X_mats"]; Y_mats = sample_dict["Y_mats"]
        X_vec60 = sample_dict["X_vec60"]; Y_vec36 = sample_dict["Y_vec36"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 X_mats=X_mats, Y_mats=Y_mats,
                 X_vec60=X_vec60, Y_vec36=Y_vec36,
                 flatOrder=flatOrder)

# ====== 入口 ======
def show():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, DeformerMapperUI):
            w.raise_(); w.activateWindow(); return w
    ui = DeformerMapperUI()
    ui.show()
    return ui
