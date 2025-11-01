# -*- coding: utf-8 -*-
import maya.cmds as cmds
from maya import OpenMayaUI as omui
import shiboken2
from PySide2 import QtWidgets, QtCore

_WINDOW_TITLE = "Curve Follower (EP Curve) — 简洁版 + 波浪/扭曲"

def _maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(int(ptr), QtWidgets.QWidget)

def _short(name): return name.split('|')[-1]
def _safe(name):  return _short(name).replace(':', '_').replace(' ', '_')

class CurveFollowerUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(CurveFollowerUI, self).__init__(parent or _maya_main_window())
        self.setWindowTitle(_WINDOW_TITLE)
        self.setMinimumWidth(460)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self.curve = None
        self.cached_objs = []
        self._build_ui()
        self._wire()
        self._update_sel_label()

    # ---------- UI ----------
    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)

        # 选择载入
        row = QtWidgets.QHBoxLayout()
        self.sel_label = QtWidgets.QLabel("已载入：0 个物体")
        self.refresh_btn = QtWidgets.QPushButton("刷新选择并载入")
        self.clear_btn = QtWidgets.QPushButton("清空列表")
        row.addWidget(self.sel_label); row.addStretch(1)
        row.addWidget(self.refresh_btn); row.addWidget(self.clear_btn)
        v.addLayout(row)

        self.sel_list = QtWidgets.QListWidget()
        v.addWidget(self.sel_list, 1)

        # 曲线设置
        box_curve = QtWidgets.QGroupBox("曲线设置")
        form = QtWidgets.QFormLayout(box_curve)
        self.degree_spin = QtWidgets.QSpinBox(); self.degree_spin.setRange(1,5); self.degree_spin.setValue(3)
        self.name_edit = QtWidgets.QLineEdit("epPath1")
        self.create_btn = QtWidgets.QPushButton("按列表物体创建 EP 曲线")
        form.addRow("Degree：", self.degree_spin)
        form.addRow("曲线名称：", self.name_edit)
        form.addRow(self.create_btn)
        v.addWidget(box_curve)

        # 绑定设置
        box_bind = QtWidgets.QGroupBox("绑定设置（Curve → motionPath → locator → parentConstraint）")
        f2 = QtWidgets.QFormLayout(box_bind)
        self.method_combo = QtWidgets.QComboBox(); self.method_combo.addItems(["平均分布", "就近投射到曲线"])
        self.front_axis = QtWidgets.QComboBox(); self.front_axis.addItems(["X","Y","Z","-X","-Y","-Z"])
        self.up_axis = QtWidgets.QComboBox(); self.up_axis.addItems(["Y","Z","X","-Y","-Z","-X"])
        self.bank_chk = QtWidgets.QCheckBox("启用 Bank 倾斜")
        self.bank_strength = QtWidgets.QDoubleSpinBox(); self.bank_strength.setRange(-10.0,10.0); self.bank_strength.setDecimals(2); self.bank_strength.setValue(1.0)
        self.keep_offset_chk = QtWidgets.QCheckBox("保持相对偏移（Maintain Offset）"); self.keep_offset_chk.setChecked(False)
        self.bind_btn = QtWidgets.QPushButton("绑定到曲线")
        self.unbind_btn = QtWidgets.QPushButton("解绑")
        self.pick_curve_btn = QtWidgets.QPushButton("用当前选中曲线作为控制曲线")
        f2.addRow("参数分配：", self.method_combo)
        f2.addRow("前向轴：", self.front_axis)
        f2.addRow("上向轴：", self.up_axis)
        f2.addRow(self.bank_chk)
        f2.addRow("Bank 强度（bankScale）：", self.bank_strength)
        f2.addRow(self.keep_offset_chk)
        f2.addRow(self.bind_btn)
        f2.addRow(self.unbind_btn)
        f2.addRow(self.pick_curve_btn)
        v.addWidget(box_bind)

        # 曲线形变（两个按钮）
        deform = QtWidgets.QGroupBox("曲线形变（作用于当前控制曲线）")
        h = QtWidgets.QHBoxLayout(deform)
        self.wave_btn = QtWidgets.QPushButton("创建波浪 (sine)")
        self.twist_btn = QtWidgets.QPushButton("创建扭曲 (twist)")
        h.addWidget(self.wave_btn); h.addWidget(self.twist_btn)
        v.addWidget(deform)

        self.status = QtWidgets.QLabel("")
        v.addWidget(self.status)

    def _wire(self):
        self.refresh_btn.clicked.connect(self._refresh_and_load_selection)
        self.clear_btn.clicked.connect(self._clear_list)
        self.create_btn.clicked.connect(self._create_ep_curve_from_list)
        self.bind_btn.clicked.connect(self._bind_list_to_curve)
        self.unbind_btn.clicked.connect(self._unbind_list_from_curve)
        self.pick_curve_btn.clicked.connect(self._pick_curve_from_selection)
        self.wave_btn.clicked.connect(lambda: self._create_curve_deform("sine"))
        self.twist_btn.clicked.connect(lambda: self._create_curve_deform("twist"))

    # ---------- Helpers ----------
    def _msg(self, text):
        self.status.setText(text)
        try: cmds.inViewMessage(amg=text, pos='midCenter', fade=True)
        except Exception: pass

    def _update_sel_label(self):
        self.sel_label.setText("已载入：{} 个物体".format(len(self.cached_objs)))

    def _get_selection_transforms(self):
        return cmds.ls(sl=True, long=True, type="transform") or []

    def _refresh_and_load_selection(self):
        current = self._get_selection_transforms()
        for n in current:
            if n not in self.cached_objs:
                self.cached_objs.append(n)
                item = QtWidgets.QListWidgetItem(_short(n))
                item.setData(QtCore.Qt.UserRole, n)
                self.sel_list.addItem(item)
        self._update_sel_label(); self._msg("载入完成。")

    def _clear_list(self):
        self.cached_objs = []; self.sel_list.clear(); self._update_sel_label()

    def _list_objects_existing(self):
        kept = []
        for i in range(self.sel_list.count()):
            it = self.sel_list.item(i)
            p = it.data(QtCore.Qt.UserRole)
            if cmds.objExists(p): kept.append(p)
        self.cached_objs = kept; self._update_sel_label(); return kept

    def _get_axes(self):
        axis_map = {"X":0, "Y":1, "Z":2, "-X":3, "-Y":4, "-Z":5}
        return axis_map[self.front_axis.currentText()], axis_map[self.up_axis.currentText()]

    def _ensure_curve_shape(self):
        if not self.curve or not cmds.objExists(self.curve): return None
        shapes = cmds.listRelatives(self.curve, shapes=True, fullPath=True) or []
        for s in shapes:
            if cmds.nodeType(s) == "nurbsCurve": return s
        return None

    # ---------- 曲线 ----------
    def _pick_curve_from_selection(self):
        sel = cmds.ls(sl=True, long=True) or []
        for s in sel:
            shp = cmds.listRelatives(s, shapes=True, fullPath=True) or []
            if any(cmds.nodeType(x)=="nurbsCurve" for x in shp):
                self.curve = s
                self._msg("已设置控制曲线：{}".format(s))
                return
        self._msg("请选择含 nurbsCurve 的 transform。")

    def _create_ep_curve_from_list(self):
        objs = self._list_objects_existing()
        if len(objs) < 3:
            self._msg("至少需要 3 个物体。"); return
        points = [cmds.xform(o, q=True, ws=True, t=True) for o in objs]
        deg = min(self.degree_spin.value(), len(points)-1)
        self.curve = cmds.curve(ep=points, d=deg, name=self.name_edit.text() or "epCurve1")
        cmds.makeIdentity(self.curve, apply=True, t=True, r=True, s=True, n=False)
        self._msg("曲线创建完成：{}".format(self.curve))

    # ---------- 形变辅助 ----------
    def _curve_endpoints_world(self, curve_shape):
        """返回曲线在 min/maxU 处的世界坐标两点"""
        minV = cmds.getAttr(curve_shape + ".minValue")
        maxV = cmds.getAttr(curve_shape + ".maxValue")
        p0 = cmds.pointOnCurve(curve_shape, pr=minV, p=True)
        p1 = cmds.pointOnCurve(curve_shape, pr=maxV, p=True)
        return p0, p1

    # ---------- 形变（sine / twist） ----------
    def _create_curve_deform(self, deform_type):
        """创建 sine / twist：句柄对齐曲线方向；扩大影响范围；不挂到曲线下面"""
        if not self.curve or not cmds.objExists(self.curve):
            self._msg("请先选择或创建控制曲线。"); return
        safe = _safe(self.curve)

        # 避免重复创建
        existing = cmds.listConnections(self.curve, type=deform_type) or []
        if existing:
            self._msg("曲线上已存在 {} 形变：{}".format(deform_type, existing[0])); return

        # 创建非线性形变（作用对象：曲线transform）
        handle, node = cmds.nonLinear(self.curve, type=deform_type, name="{}_{}".format(safe, deform_type))

        # 句柄放置到旁系组（不 parent 到曲线）
        grp_name = "{}_deformers_GRP".format(safe)
        if not cmds.objExists(grp_name):
            parent_of_curve = cmds.listRelatives(self.curve, p=True) or []
            grp_name = cmds.createNode("transform", name=grp_name, p=parent_of_curve[0] if parent_of_curve else None)
        try:
            cmds.parent(handle, grp_name)
        except Exception:
            pass
        cmds.setAttr(handle + ".translate", 0,0,0, type="double3")
        cmds.setAttr(handle + ".rotate", 0,0,0, type="double3")
        cmds.setAttr(handle + ".scale", 1,1,1, type="double3")

        # 对齐：句柄朝向沿曲线方向（首→尾），位置放中点
        curve_shape = self._ensure_curve_shape()
        p0, p1 = self._curve_endpoints_world(curve_shape)
        mid = [(p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5]
        cmds.xform(handle, ws=True, t=mid)
        loc_a = cmds.spaceLocator(name="{}_aimA_tmp".format(safe))[0]
        loc_b = cmds.spaceLocator(name="{}_aimB_tmp".format(safe))[0]
        cmds.xform(loc_a, ws=True, t=p0); cmds.xform(loc_b, ws=True, t=p1)
        ac = cmds.aimConstraint(loc_b, handle, aimVector=(0,1,0), upVector=(0,0,1),
                                worldUpType="object", worldUpObject=loc_a)[0]
        cmds.delete(ac, loc_a, loc_b)

        # 默认参数 + 扩大范围（确保可见效果）
        if deform_type == "twist":
            cmds.setAttr(node + ".endAngle", 45.0)
            cmds.setAttr(node + ".lowBound", -10.0)
            cmds.setAttr(node + ".highBound", 10.0)
            cmds.setAttr(node + ".envelope", 1.0)
        elif deform_type == "sine":
            cmds.setAttr(node + ".amplitude", 1.0)
            cmds.setAttr(node + ".wavelength", 5.0)
            cmds.setAttr(node + ".offset", 0.0)
            cmds.setAttr(node + ".dropoff", 0.0)
            cmds.setAttr(node + ".lowBound", -10.0)
            cmds.setAttr(node + ".highBound", 10.0)
            cmds.setAttr(node + ".envelope", 1.0)

        self._msg("已创建 {} 形变：{}（句柄在 {}，可旋转/移动调整方向与范围）".format(deform_type, node, grp_name))

    # ---------- 绑定 ----------
    def _bind_list_to_curve(self):
        objs = self._list_objects_existing()
        if not objs: self._msg("列表为空。"); return
        curve_shape = self._ensure_curve_shape()
        if not curve_shape: self._msg("请先创建或选择曲线。"); return

        method = self.method_combo.currentText()
        front, up = self._get_axes()
        bank_on = self.bank_chk.isChecked()
        bank_strength = float(self.bank_strength.value())
        keep_offset = self.keep_offset_chk.isChecked()

        minV, maxV = cmds.getAttr(curve_shape + ".minValue"), cmds.getAttr(curve_shape + ".maxValue")
        span = max(1e-6, maxV - minV)
        n = len(objs)

        for i, o in enumerate(objs):
            safe = _safe(o)
            loc = cmds.createNode("transform", name=f"{safe}_mpLoc")
            mp = cmds.createNode("motionPath", name=f"{safe}_mp")
            cmds.connectAttr(curve_shape + ".worldSpace[0]", mp + ".geometryPath", f=True)
            cmds.setAttr(mp + ".fractionMode", 1)
            cmds.setAttr(mp + ".follow", 1)
            cmds.setAttr(mp + ".frontAxis", front)
            cmds.setAttr(mp + ".upAxis", up)
            cmds.setAttr(mp + ".bank", 1 if bank_on else 0)
            cmds.setAttr(mp + ".bankScale", bank_strength)
            cmds.setAttr(mp + ".worldUpType", 0)

            # u 值
            if method == "平均分布":
                u = 0.0 if n==1 else i/float(n-1)
            else:
                npc = cmds.createNode("nearestPointOnCurve", name=f"{safe}_npc")
                cmds.connectAttr(curve_shape + ".worldSpace[0]", npc + ".inputCurve", f=True)
                pos = cmds.xform(o, q=True, ws=True, t=True)
                cmds.setAttr(npc + ".inPosition", *pos, type="double3")
                param = cmds.getAttr(npc + ".parameter")
                cmds.delete(npc)
                u = (param - minV)/span
            cmds.setAttr(mp + ".uValue", max(0,min(1,u)))

            cmds.connectAttr(mp + ".allCoordinates", loc + ".translate", f=True)
            cmds.connectAttr(mp + ".rotate", loc + ".rotate", f=True)
            cmds.parentConstraint(loc, o, mo=keep_offset, weight=1.0)
        self._msg("绑定完成：{} 个物体，Bank强度={}".format(len(objs), bank_strength))

    def _unbind_list_from_curve(self):
        objs = self._list_objects_existing()
        for o in objs:
            cons = cmds.listConnections(o, s=True, d=False, type="parentConstraint") or []
            for c in cons:
                targets = cmds.parentConstraint(c, q=True, tl=True) or []
                for t in targets:
                    if "_mpLoc" in t and cmds.objExists(t):
                        mps = cmds.listConnections(t, s=True, d=False, type="motionPath") or []
                        for mp in mps:
                            if cmds.objExists(mp): cmds.delete(mp)
                        if cmds.objExists(t): cmds.delete(t)
                if cmds.objExists(c): cmds.delete(c)
        self._msg("解绑完成。")

# 入口
_ui=None
def show_curve_tool():
    global _ui
    try:
        if _ui and _ui.isVisible(): _ui.close()
    except: pass
    _ui=CurveFollowerUI(); _ui.show()
    return _ui

try: show_curve_tool()
except Exception as e:
    import traceback; traceback.print_exc()
    cmds.warning(str(e))
