from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsView
COLORS = [(255,75,75),(57,189,248),(75,220,120),(255,196,65),(190,100,255),(255,105,190),(75,230,220)]
class AnnotationView(QGraphicsView):
    changed = Signal()
    selection_requested = Signal(int)
    status = Signal(str)
    box_changed = Signal(object)
    masks_changed = Signal()
    instance_finished = Signal(int)
    small_mask_finished = Signal(int,int)
    tool_reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self)); self.setBackgroundBrush(QColor("#090d14"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.image_item = None; self.array = None; self.masks = []; self.selected = -1; self.show_labels = True
        self.mode = "idle"; self.brush_size = 20; self.polygon = []; self.preview = None; self.drawing = False
        self._middle_panning=False; self._middle_pan_pos=None
        self.sam_box = None; self.box_start = None
        self.exemplar_mask = None
        self._overlay_cache=None; self._overlay_signature=None; self._editing_background=None; self.overlay_item=None; self.active_overlay_item=None
        self._background_key=None; self._hit_map=None; self._hit_signature=None; self._bbox_cache={}
        self._live_redraw_pending=False; self._live_redraw_timer=QTimer(self); self._live_redraw_timer.setSingleShot(True)
        self._live_redraw_timer.setInterval(33); self._live_redraw_timer.timeout.connect(self._flush_live_redraw)
        self.new_mask_metadata = {"category_id":1,"noun_phrase":"particle","annotation_source":"manual"}
        self._update_tool_cursor()

    def set_image(self, array: np.ndarray, masks: list[dict]):
        self.scene().clear(); self.array = array; self.masks = masks; self.selected = -1; self.polygon = []; self.overlay_item=None; self.active_overlay_item=None
        self._overlay_cache=None; self._overlay_signature=None; self._editing_background=None; self._background_key=None
        self._hit_map=None; self._hit_signature=None; self._bbox_cache={}
        h,w=array.shape[:2]
        for entry in masks:
            bbox=entry.get("bbox")
            if bbox and len(bbox)>=4 and not entry.get("_geometry_dirty",False):
                x,y,bw,bh=[int(round(float(value))) for value in bbox[:4]]
                self._bbox_cache[id(entry["mask"])]=(max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)) if bw>0 and bh>0 else None
        rgb = np.ascontiguousarray(array[..., :3]); h,w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.image_item = self.scene().addPixmap(QPixmap.fromImage(qimg)); self.image_item.setZValue(0)
        self.redraw(); self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def redraw(self):
        for item in list(self.scene().items()):
            if item is not self.image_item and item is not self.overlay_item and item is not self.active_overlay_item: self.scene().removeItem(item)
        if self.array is None: return
        h,w=self.array.shape[:2]
        # Keep one composite of all instances. Selection changes only add a
        # highlight and therefore no longer rebuild hundreds of SAM3 masks.
        signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
        if signature!=self._overlay_signature:
            self._overlay_cache=self._compose_overlay(); self._overlay_signature=signature
        editing=self.drawing and self._editing_background is not None
        background=self._editing_background if editing else self._overlay_cache
        background_key=("editing",self.selected,signature) if editing else ("base",signature)
        if background_key!=self._background_key:
            rgba=np.ascontiguousarray(background); qimg=QImage(rgba.data,w,h,rgba.strides[0],QImage.Format_RGBA8888).copy(); pixmap=QPixmap.fromImage(qimg)
            if self.overlay_item is None:
                self.overlay_item=self.scene().addPixmap(pixmap); self.overlay_item.setZValue(2)
            else:self.overlay_item.setPixmap(pixmap)
            self._background_key=background_key
        if self.overlay_item is not None:self.overlay_item.setVisible(self.show_labels)
        if self.show_labels:self._redraw_active_overlay()
        elif self.active_overlay_item is not None:self.active_overlay_item.setVisible(False)
        if self.show_labels and self.exemplar_mask is not None:
            contours,_=cv2.findContours(self.exemplar_mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            path=QPainterPath()
            for contour in contours:
                points=contour.reshape(-1,2)
                if not len(points):continue
                path.moveTo(float(points[0,0]),float(points[0,1]))
                for point in points[1:]:path.lineTo(float(point[0]),float(point[1]))
                path.closeSubpath()
            item=QGraphicsPathItem(path); item.setPen(QPen(QColor(190,60,255),4)); item.setZValue(4); self.scene().addItem(item)
        if self.polygon:
            path = QPainterPath(QPointF(*self.polygon[0]))
            for p in self.polygon[1:]: path.lineTo(QPointF(*p))
            self.preview = QGraphicsPathItem(path); self.preview.setPen(QPen(QColor("#ffff00"),2)); self.preview.setZValue(4); self.scene().addItem(self.preview)
        if self.sam_box:
            x0,y0,x1,y1=self.sam_box; path=QPainterPath(); path.addRect(float(x0),float(y0),float(x1-x0),float(y1-y0))
            item=QGraphicsPathItem(path); item.setPen(QPen(QColor("#ffff00"),3)); item.setZValue(5); self.scene().addItem(item)

    def _compose_overlay(self, exclude=-1):
        h,w=self.array.shape[:2]; overlay=np.zeros((h,w,4),dtype=np.uint8)
        for idx,entry in enumerate(self.masks):
            if idx==exclude:continue
            mask=entry["mask"]; box=self._mask_bbox(mask)
            if box is None:continue
            x0,y0,x1,y1=box; crop=mask[y0:y1,x0:x1]; region=overlay[y0:y1,x0:x1]
            region[crop,:3]=COLORS[idx%len(COLORS)]; region[crop,3]=100
        return overlay

    def _editing_overlay(self,selected):
        # Start from the already composed background and repair only the
        # selected instance's bounding box. This preserves overlapping masks
        # without recomposing every full-resolution SAM3 mask.
        if self._overlay_cache is None:return self._compose_overlay(selected)
        active=self.masks[selected]["mask"]; box=self._mask_bbox(active)
        if box is None:return self._overlay_cache
        background=self._overlay_cache.copy(); x0,y0,x1,y1=box
        active_crop=active[y0:y1,x0:x1]; region=background[y0:y1,x0:x1]; region[active_crop]=0
        for idx,entry in enumerate(self.masks):
            if idx==selected:continue
            overlap=entry["mask"][y0:y1,x0:x1] & active_crop
            if overlap.any():region[overlap,:3]=COLORS[idx%len(COLORS)]; region[overlap,3]=100
        return background

    def _commit_editing_overlay(self,selected):
        if self._editing_background is None:return
        background=self._editing_background.copy(); active=self.masks[selected]["mask"]; box=self._mask_bbox(active)
        if box is not None:
            x0,y0,x1,y1=box; active_crop=active[y0:y1,x0:x1]; region=background[y0:y1,x0:x1]
            region[active_crop,:3]=COLORS[selected%len(COLORS)]; region[active_crop,3]=100
            # Masks later in the list are visually on top; restore only their
            # overlap with the edited instance.
            for idx in range(selected+1,len(self.masks)):
                overlap=self.masks[idx]["mask"][y0:y1,x0:x1] & active_crop
                if overlap.any():region[overlap,:3]=COLORS[idx%len(COLORS)]; region[overlap,3]=100
        h,w=self.array.shape[:2]
        self._overlay_cache=background; self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks))

    def _mask_bbox(self,mask):
        key=id(mask)
        if key not in self._bbox_cache:
            ys,xs=np.nonzero(mask); self._bbox_cache[key]=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1) if len(xs) else None
        return self._bbox_cache[key]

    def _redraw_active_overlay(self):
        if self.active_overlay_item is not None:
            self.active_overlay_item.setVisible(False)
        if self.selected<0 or self.selected>=len(self.masks):return
        mask=self.masks[self.selected]["mask"]; box=self._mask_bbox(mask)
        if box is None:return
        x0,y0,x1,y1=box; crop=mask[y0:y1,x0:x1]; overlay=np.zeros((y1-y0,x1-x0,4),np.uint8)
        overlay[crop,:3]=COLORS[self.selected%len(COLORS)]; overlay[crop,3]=155
        if not self.drawing:
            contours,_=cv2.findContours(crop.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(*COLORS[self.selected%len(COLORS)],255),2)
        rgba=np.ascontiguousarray(overlay); qimg=QImage(rgba.data,x1-x0,y1-y0,rgba.strides[0],QImage.Format_RGBA8888).copy(); pixmap=QPixmap.fromImage(qimg)
        if self.active_overlay_item is None:
            self.active_overlay_item=self.scene().addPixmap(pixmap); self.active_overlay_item.setZValue(3)
        else:self.active_overlay_item.setPixmap(pixmap)
        self.active_overlay_item.setPos(x0,y0); self.active_overlay_item.setVisible(True)

    def _ensure_hit_map(self):
        h,w=self.array.shape[:2]; signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
        if signature!=self._hit_signature:
            hit_map=np.full((h,w),-1,np.int32)
            for idx,entry in enumerate(self.masks):hit_map[entry["mask"]]=idx
            self._hit_map=hit_map; self._hit_signature=signature

    def set_mode(self, mode: str):
        self.mode = mode; self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "pan" else QGraphicsView.NoDrag)
        self._update_tool_cursor()
        self.status.emit({"idle":"Select an instance","pan":"Pan / zoom","select":"Select an instance","polygon":"Click to add points; right-click or Enter to finish","brush":"Brush: paint; Enter finishes the instance","erase":"Erase selected instance","sam_box":"Draw box"}[mode])

    def set_labels_visible(self,visible):
        self.show_labels=bool(visible); self.redraw()

    def set_brush_size(self, size: int):
        self.brush_size=int(size); self._update_tool_cursor()

    def _update_tool_cursor(self):
        if self.mode=="pan":cursor=QCursor(Qt.OpenHandCursor)
        elif self.mode in {"idle","select"}:cursor=QCursor(Qt.ArrowCursor)
        elif self.mode=="polygon":cursor=QCursor(Qt.CrossCursor)
        elif self.mode=="sam_box":
            canvas=25; center=canvas//2; pixmap=QPixmap(canvas,canvas); pixmap.fill(Qt.transparent)
            painter=QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing); painter.setPen(QPen(QColor("#55d8ff"),2))
            painter.drawRect(3,3,canvas-7,canvas-7); painter.drawLine(center,0,center,canvas-1); painter.drawLine(0,center,canvas-1,center); painter.end()
            cursor=QCursor(pixmap,center,center)
        elif self.mode in {"brush","erase"}:
            scale=max(abs(self.transform().m11()),0.05); diameter=max(7,min(61,int(round(self.brush_size*scale))))
            canvas=max(17,diameter+6); pixmap=QPixmap(canvas,canvas); pixmap.fill(Qt.transparent)
            painter=QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
            color=QColor("#f4f8ff") if self.mode=="brush" else QColor("#ff5b5b")
            painter.setPen(QPen(color,2)); margin=3; painter.drawEllipse(margin,margin,canvas-2*margin-1,canvas-2*margin-1)
            center=canvas//2
            if self.mode=="brush":painter.drawPoint(center,center)
            else:
                arm=max(3,min(9,diameter//3)); painter.drawLine(center-arm,center-arm,center+arm,center+arm); painter.drawLine(center-arm,center+arm,center+arm,center-arm)
            painter.end(); cursor=QCursor(pixmap,center,center)
        else:cursor=QCursor(Qt.ArrowCursor)
        self.viewport().setCursor(cursor)

    def wheelEvent(self, event):
        if not event.angleDelta().y():return super().wheelEvent(event)
        cursor=event.position().toPoint(); anchor_before=self.mapToScene(cursor)
        current=abs(self.transform().m11()); requested=1.2 if event.angleDelta().y()>0 else 1/1.2
        target=max(0.05,min(40.0,current*requested)); factor=target/current if current else 1.0
        previous_anchor=self.transformationAnchor(); self.setTransformationAnchor(QGraphicsView.NoAnchor); self.scale(factor,factor)
        anchor_after=self.mapToScene(cursor); delta=anchor_after-anchor_before; self.translate(delta.x(),delta.y())
        self.setTransformationAnchor(previous_anchor); self._update_tool_cursor(); event.accept()

    def image_pos(self, event):
        if self.image_item is None: return None
        p = self.image_item.mapFromScene(self.mapToScene(event.position().toPoint()))
        x,y = int(round(p.x())),int(round(p.y()))
        h,w = self.array.shape[:2]
        return (x,y) if 0 <= x < w and 0 <= y < h else None

    def mousePressEvent(self, event):
        if event.button()==Qt.MiddleButton:
            self._middle_panning=True; self._middle_pan_pos=event.position(); self.viewport().setCursor(QCursor(Qt.ClosedHandCursor)); event.accept(); return
        if event.button()==Qt.RightButton and self.mode in {"brush","erase"}:
            self.finish_current_instance(); event.accept(); return
        pos = self.image_pos(event)
        if pos is None: return super().mousePressEvent(event)
        x,y = pos
        self._ensure_hit_map(); hit=int(self._hit_map[y,x])
        if self.mode=="idle":
            if hit>=0:self.selection_requested.emit(hit)
            else:self.selection_requested.emit(-1)
            return super().mousePressEvent(event)
        # Existing annotations are selectable directly in every interactive
        # mode.  This keeps the image overlay, instance list and label fields
        # synchronized before a user edits or deletes the selected instance.
        if event.button() == Qt.LeftButton and hit>=0 and self.mode in {"pan","select","brush","erase"}:
            if self.selected!=hit:self.selection_requested.emit(hit)
            if self.mode=="pan":return super().mousePressEvent(event)
        elif self.mode=="pan":return super().mousePressEvent(event)
        if self.mode == "select":
            if hit>=0: self.selection_requested.emit(hit)
            else: self.selected=-1; self.selection_requested.emit(-1); self.redraw()
        elif self.mode == "polygon":
            if event.button() == Qt.RightButton: self.finish_polygon()
            else: self.polygon.append((x,y)); self.redraw()
        elif self.mode in {"brush","erase"}:
            if self.selected < 0 and self.mode == "erase": self.status.emit("Select an instance first"); return
            created = False
            if self.selected < 0 and self.mode == "brush":
                self.changed.emit(); h,w=self.array.shape[:2]
                item=dict(self.new_mask_metadata); item.update({"id":0,"mask":np.zeros((h,w),bool)})
                self.masks.append(item); self.selected=len(self.masks)-1
                # The new mask is empty, so the existing composite remains
                # valid even though the mask list signature has changed.
                self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
                self.selection_requested.emit(self.selected); self.masks_changed.emit(); created=True
            self._editing_background=self._overlay_cache if created else self._editing_overlay(self.selected)
            self._background_key=None; self.drawing = True
            if not created:self.changed.emit()
            self.paint_at(x,y)
        elif self.mode == "sam_box" and event.button()==Qt.LeftButton:
            self.box_start=(x,y); self.sam_box=(x,y,x,y); self.drawing=True; self.redraw()

    def mouseMoveEvent(self,event):
        if self._middle_panning and self._middle_pan_pos is not None:
            delta=event.position()-self._middle_pan_pos; self._middle_pan_pos=event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-int(round(delta.x())))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-int(round(delta.y())))
            event.accept(); return
        if self.drawing and self.mode in {"brush","erase"}:
            pos=self.image_pos(event)
            if pos: self.paint_at(*pos)
        elif self.drawing and self.mode=="sam_box":
            pos=self.image_pos(event)
            if pos:
                x0,y0=self.box_start; x1,y1=pos; self.sam_box=(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)); self.redraw()
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MiddleButton and self._middle_panning:
            self._middle_panning=False; self._middle_pan_pos=None; self._update_tool_cursor(); event.accept(); return
        was_box=self.drawing and self.mode=="sam_box" and event.button()==Qt.LeftButton
        if self.mode in {"brush","erase"}:
            if self.drawing and 0<=self.selected<len(self.masks):self._commit_editing_overlay(self.selected)
            self.drawing=False;self._live_redraw_timer.stop();self._live_redraw_pending=False;self._editing_background=None;self._background_key=None;self._hit_signature=None;self.redraw()
        else:self.drawing=False
        if was_box:
            pos=self.image_pos(event)
            if pos:
                x0,y0=self.box_start; x1,y1=pos
                self.sam_box=(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)); self.redraw()
            self.box_changed.emit(self.sam_box)
        super().mouseReleaseEvent(event)

    def paint_at(self,x,y):
        self.masks[self.selected]["_geometry_dirty"]=True
        mask=self.masks[self.selected]["mask"]; radius=max(1,self.brush_size//2); h,w=mask.shape
        x0=max(0,x-radius);x1=min(w,x+radius+1);y0=max(0,y-radius);y1=min(h,y+radius+1)
        yy,xx=np.ogrid[y0-y:y1-y,x0-x:x1-x]; disk=xx*xx+yy*yy<=radius*radius
        region=mask[y0:y1,x0:x1];region[disk]=self.mode=="brush"
        if self.mode=="brush":
            old=self._bbox_cache.get(id(mask)); self._bbox_cache[id(mask)]=(x0,y0,x1,y1) if old is None else (min(old[0],x0),min(old[1],y0),max(old[2],x1),max(old[3],y1))
        if not self._live_redraw_pending:
            self._live_redraw_pending=True;self._live_redraw_timer.start()

    def _flush_live_redraw(self):
        self._live_redraw_pending=False;self.redraw()

    def finish_polygon(self):
        if len(self.polygon) < 3: self.polygon=[]; self.redraw(); return
        self.changed.emit(); h,w=self.array.shape[:2]; mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[np.asarray(self.polygon,np.int32)],1)
        item=dict(self.new_mask_metadata); item.update({"id":0,"mask":mask.astype(bool)})
        self.masks.append(item); self.polygon=[]; self.selection_requested.emit(len(self.masks)-1); self.masks_changed.emit(); self.redraw()

    def clear_mask_selection(self):
        self.selected=-1; self.selection_requested.emit(-1); self.redraw()

    def detach_selected_mask(self):
        if not (0<=self.selected<len(self.masks)) or self.array is None:return
        old_mask=self.masks[self.selected]["mask"]; new_mask=old_mask.copy(); self.masks[self.selected]["mask"]=new_mask
        if id(old_mask) in self._bbox_cache:self._bbox_cache[id(new_mask)]=self._bbox_cache[id(old_mask)]
        h,w=self.array.shape[:2]; self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks)); self._hit_signature=None

    def finish_current_instance(self):
        if self.mode not in {"brush","erase"} or not (0<=self.selected<len(self.masks)):return
        finished=self.selected; finished_item=self.masks[finished]; area=int(finished_item["mask"].sum())
        self.masks_changed.emit(); self.clear_mask_selection(); self.status.emit("Instance finished; paint to create another")
        if area<10:self.small_mask_finished.emit(finished,area)
        remaining=next((i for i,item in enumerate(self.masks) if item is finished_item),-1)
        if self.mode=="brush" and remaining>=0:self.instance_finished.emit(remaining)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape:
            self.polygon=[]; self.drawing=False; self._live_redraw_timer.stop(); self._live_redraw_pending=False; self._editing_background=None; self._overlay_signature=None; self._background_key=None; self._hit_signature=None; self.redraw(); self.tool_reset_requested.emit(); return
        if event.key() in (Qt.Key_Return,Qt.Key_Enter) and self.mode=="polygon":
            self.finish_polygon(); finished=self.selected
            if 0<=finished<len(self.masks) and int(self.masks[finished]["mask"].sum())<10:self.small_mask_finished.emit(finished,int(self.masks[finished]["mask"].sum()))
            self.clear_mask_selection()
        elif event.key() in (Qt.Key_Return,Qt.Key_Enter) and self.mode in {"brush","erase"}:self.finish_current_instance()
        else: super().keyPressEvent(event)
